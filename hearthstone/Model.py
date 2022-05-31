import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightnTransfomer import rightTransformerBlock
from Multihead_Combination import MultiHeadedCombination
from Embedding import Embedding
from TreeConvGen import TreeConvGen
from Multihead_Attention import MultiHeadedAttention
from gelu import GELU
from LayerNorm import LayerNorm
from decodeTrans import decodeTransformerBlock
from gcnnnormal import GCNNM
from postionEmbedding import PositionalEmbedding
class TreeAttEncoder(nn.Module):
    def __init__(self, args):
        super(TreeAttEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])
        self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(3)])


    def forward(self, input_code, input_codechar, inputAd):
        codemask = torch.gt(input_code, 0)
        charEm = self.char_embedding(input_codechar)
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        #print(charEm.shape)
        x = self.token_embedding(input_code.long())
        for trans in self.transformerBlocksTree:
            x = trans.forward(x, codemask, charEm, inputAd, True)
        for trans in self.transformerBlocks:
            x = trans.forward(x, codemask, charEm)
        return x

class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.token_embedding = Embedding(args.Nl_Vocsize, self.embedding_size)
        '''self.transformerBlocksTree = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])'''


    def forward(self, input_nl, input_nlchar):
        nlmask = torch.gt(input_nl, 0)
        charEm = self.char_embedding(input_nlchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        x = self.token_embedding(input_nl.long())
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, charEm)
        return x, nlmask
class CopyNet(nn.Module):
    def __init__(self, args):
        super(CopyNet, self).__init__()
        self.embedding_size = args.embedding_size
        self.LinearSource = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearTarget = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.LinearRes = nn.Linear(self.embedding_size, 1)
        self.LinearProb = nn.Linear(self.embedding_size, 2)
    def forward(self, source, traget):
        sourceLinear = self.LinearSource(source)
        targetLinear = self.LinearTarget(traget)
        genP = self.LinearRes(F.tanh(sourceLinear.unsqueeze(1) + targetLinear.unsqueeze(2))).squeeze(-1)
        prob = F.softmax(self.LinearProb(traget), dim=-1)#.squeeze(-1))
        return genP, prob
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.code_len = args.CodeLen
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, args.WoLen))
        self.path_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.rule_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 2))
        self.depth_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 40))
        self.resLen = args.rulenum - args.NlLen
        self.encodeTransformerBlock = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(9)])
        self.decodeTransformerBlocksP = nn.ModuleList(
            [decodeTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(2)])
        self.finalLinear = nn.Linear(self.embedding_size, 2048)
        self.resLinear = nn.Linear(2048, self.resLen)
        self.rule_token_embedding = Embedding(args.Code_Vocsize, self.embedding_size)
        self.rule_embedding = nn.Embedding(args.rulenum, self.embedding_size)
        self.encoder = NlEncoder(args)
        self.layernorm = LayerNorm(self.embedding_size)
        self.activate = GELU()
        self.copy = CopyNet(args)
        self.copy2 = CopyNet(args)
        self.dropout = nn.Dropout(p=0.1)
        self.depthembedding = nn.Embedding(40, self.embedding_size, padding_idx=0)
        self.gcnnm = GCNNM(self.embedding_size)
        self.position = PositionalEmbedding(self.embedding_size)
    def getBleu(self, losses, ngram):
        bleuloss = F.max_pool1d(losses.unsqueeze(1), ngram, 1).squeeze(1)
        bleuloss = torch.sum(bleuloss, dim=-1)  
        return bleuloss
    def forward(self, inputnl, inputnlchar, inputrule, inputruleparent, inputrulechild, inputParent, inputParentPath, inputdepth, tmpf, tmpc, tmpindex, rulead, antimask, inputRes=None, mode="train"):
        selfmask = antimask
        #selfmask = antimask.unsqueeze(0).repeat(inputtype.size(0), 1, 1).unsqueeze(1)
        #admask = admask.unsqueeze(0).repeat(inputtype.size(0), 1, 1).float()
        rulemask = torch.gt(inputrule, 0)
        inputParent = inputParent.float()
        #encode_nl
        nlencode, nlmask = self.encoder(inputnl, inputnlchar)
        #encode_rule
        childEm = self.rule_token_embedding(tmpc)
        childEm = self.conv(childEm.permute(0, 3, 1, 2))
        childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        childEm = self.layernorm(childEm)
        fatherEm = self.rule_token_embedding(tmpf)
        ruleEmCom = self.rule_conv(torch.stack([fatherEm, childEm], dim=-2).permute(0, 3, 1, 2))
        ruleEmCom = self.layernorm(ruleEmCom.permute(0, 2, 3, 1).squeeze(dim=-2))
        x = self.rule_embedding(tmpindex[0])
        # for i in range(9):
        #     x = self.gcnnm(x, rulead[0], ruleEmCom[0]).view(self.resLen, self.embedding_size)
        ruleEm = F.embedding(inputrule, x)#self.rule_embedding(inputrule)
        ruleselect = x
        depthEm = self.depthembedding(inputdepth.long())
        depthEm = self.depth_conv(depthEm.permute(0, 3, 1, 2))
        depthEm = depthEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        depthEm = self.layernorm(depthEm)
        Ppath = self.rule_token_embedding(inputrulechild)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ruleEm + self.position(inputrule))
        for trans in self.encodeTransformerBlock:
            x = trans(x, selfmask, nlencode, nlmask, ppathEm, inputParent)
        decode = x
        #ppath
        Ppath = self.rule_token_embedding(inputParentPath)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = self.dropout(ppathEm + self.position(inputRes))
        for trans in self.decodeTransformerBlocksP:
            x = trans(x, rulemask, decode, antimask, nlencode, nlmask)
        decode = x
        genP1, _ = self.copy2(ruleselect.unsqueeze(0), decode)
        #resSoftmax = F.softmax(genP, dim=-1)
        genP, prob = self.copy(nlencode, decode)
        copymask = nlmask.unsqueeze(1).repeat(1, inputrule.size(1), 1)
        genP = genP.masked_fill(copymask==0, -1e9)       
        genP = torch.cat([genP1, genP], dim=2)
        #genP = F.softmax(genP, dim=-1)

        #x = self.finalLinear(decode)
        #x = self.activate(x)
        #x = self.resLinear(x)
        #resSoftmax = F.softmax(x, dim=-1)

        #resSoftmax = resSoftmax * prob[:,:,0].unsqueeze(-1)
        #genP = genP * prob[:,:,1].unsqueeze(-1)
        resSoftmax = F.softmax(genP, dim=-1)#torch.cat([resSoftmax, genP], -1)
        if mode != "train":
            return resSoftmax
        resmask = torch.gt(inputRes, 0)
        loss = -torch.log(torch.gather(resSoftmax, -1, inputRes.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = torch.sum(resmask, dim=-1).float()
        totalloss = torch.mean(loss, dim=-1) * self.code_len / resTruelen
        totalloss = totalloss# + (self.getBleu(loss, 2) + self.getBleu(loss, 3) + self.getBleu(loss, 4)) / resTruelen
        #totalloss = torch.mean(totalloss)
        return totalloss, resSoftmax



class JointEmbber(nn.Module):
    def __init__(self, args):
        super(JointEmbber, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = TreeAttEncoder(args)
        self.margin = args.margin
        self.nlEncoder = NlEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.poolConvcode = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.maxPoolcode = nn.MaxPool1d(args.CodeLen)
    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim
    def nlencoding(self, inputnl, inputnlchar):
        nl = self.nlEncoder(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl
    def codeencoding(self, inputcode, inputcodechar, ad):
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        return code
    def forward(self, inputnl, inputnlchar, inputcode, inputcodechar, ad, inputcodeneg, inputcodenegchar, adneg):
        nl = self.nlEncoder(inputnl, inputnlchar)
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        codeneg = self.codeEncoder(inputcodeneg, inputcodenegchar, adneg)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        codeneg = self.maxPoolcode(self.poolConvcode(codeneg.permute(0, 2, 1))).squeeze(-1)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6).mean()
        return loss, good_score, bad_score










