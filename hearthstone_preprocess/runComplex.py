# -*- coding: utf-8 -*-  
sets = ["train", "test", "dev"]
import re

import ast
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
class CodeVisitor:
  def __init__(self, name, edge=""):
    self.child = []
    self.father = None
    self.edge = edge
    self.name = name
def visitNode(node):
  #print(node[0], type(node).__name__)
  rootNode = CodeVisitor(type(node).__name__)
  tmpNode = rootNode
  for x in ast.iter_fields(node):
    if str(x[0]) == "ctx":
      continue
    #print(str(x[1]))
    if str(x[0]) == "vararg" or str(x[0]) == "kwarg" or str(x[0]) == 'defaults' or str(x[0]) == "decorator_list" or str(x[0]) == "starargs" or str(x[0]) == "kwargs":
      continue
    rootNode = tmpNode
    currnode = CodeVisitor(x[0])
    rootNode.child.append(currnode)
    currnode.father = rootNode
    rootNode = currnode
    if isinstance(x[1], list):
      if len(x[1]) == 0:
        tmpnode = CodeVisitor("empty")
        rootNode.child.append(tmpnode)
        tmpnode.father = rootNode
        tmpnode.edge = x[0]
      for obj in x[1]:
        if isinstance(obj, ast.AST):
          tmpnode = visitNode(obj)
          rootNode.child.append(tmpnode)
          tmpnode.father = rootNode
          tmpnode.edge = x[0]
    elif isinstance(x[1], (int, complex)) or type(x[1]).__name__ == "float" or type(x[1]).__name__ == "long":
      #print(x[1], isinstance(x[1], bytes))
      tmpStr = str(x[1]).replace("\n", "").replace("\r", "")
      if len(tmpStr.split()) == 0:
        tmpStr = "<empty>"
      if tmpStr[-1] == "^":
        tmpStr += "<>"
      tmpnode = CodeVisitor(tmpStr)
      tmpnode.father = rootNode
      rootNode.child.append(tmpnode)
      tmpnode.edge = x[0]
    elif isinstance(x[1], str) or type(x[1]).__name__ == "unicode":
      tmpStr = x[1]
      tmpStr = tmpStr.replace("\'", "").replace(" ", "").replace("-", "").replace(":", "")
      #tmpStr = "<string>" if " " in x[1] else x[1].replace("\n", "").replace("\r", "")
      if "\t" in tmpStr:
        tmpStr = "<string>"
      if len(tmpStr.split()) == 0:
        tmpStr = "<empty>"
      if tmpStr[-1] == "^":
        tmpStr += "<>"
      '''if x[0] == 'name':
        s = "namestr"
      else:
        s = x[0]
      tmpnodef = CodeVisitor(s)
      tmpnodef.father = rootNode
      rootNode.child.append(tmpnodef)'''
      tmpnodef = rootNode
      tmpnode = CodeVisitor(tmpStr)
      tmpnode.father = tmpnodef
      tmpnodef.child.append(tmpnode)
      tmpnode.edge = x[0]
    elif isinstance(x[1], ast.AST):
      tmpnode = visitNode(x[1])
      rootNode.child.append(tmpnode)
      tmpnode.father = rootNode
      tmpnode.edge = x[0]
    elif not x[1]:
      continue
    else:
      print(type(x[1]), x[0])
      sys.exit(1)
  return tmpNode
def parseAst(codeStr):
  root_node = ast.parse(codeStr)
  #print(ast.dump(root_node))
  return visitNode(root_node)
def printTree(node):
  ans = ""
  ans += node.name + "\t"
  for x in node.child:
    ans += printTree(x)
  ans += "^" + "\t"
  return ans
def tokenize_for_bleu_eval(code):
  code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
  #code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
  code = re.sub(r'\s+', ' ', code)
  code = code.replace('"', '`')
  code = code.replace('\'', '`')
  tokens = [t for t in code.split(' ') if t]
  return tokens
grammar_file = 'py3_asdl.simplified.txt'
#asdl_text = open(grammar_file).read()
#grammar = ASDLGrammar.from_text(asdl_text)
#transition_system = Python3TransitionSystem(grammar)
maxlen = []
for x in sets:
  fout = open(x + "_hs.out", "r")
  fin = open(x + "_hs.in", "r")
  wf = open(x + ".txt", "w")
  linesin = fin.readlines()
  for i, y in enumerate(fout):
    code = y.strip().replace("§", "\n")
    try:
      parsed_ast = ast.parse(code)#parse(code, error_recovery=True, version="2.7")
      root = visitNode(parsed_ast)
      #print(ast.dump(parsed_ast))
    except Exception as e:
      print(e)
      print(code)
      assert(0)
    nl = linesin[i]
    i = nl.find("NAME_END")
    name = nl[:i].strip()
    nl = nl[i:]
    nl = tokenize_for_bleu_eval(code)[1] + " " + nl
    nl = nl.replace("<b>", " <b> ").replace("<i>", " <i> ").replace("</b>", " 在 ").replace("</i>", " 见 ").replace("+", " + ").replace("/", " / ").replace(":", "").replace("在", "</b>").replace(".", " . ").replace("(", " ( ").replace(")", " ) ").replace("见", "</i>").replace(";", " ; ").replace(",", " , ").replace("#", " # ").replace("$", " $ ")
    lst = []
    for x in nl.split():
      if "-" in x and x[0] != "-":
        lst += x.replace("-", " - ").split()
      else:
        lst.append(x)
    nl = "\t".join(lst)
    maxlen.append(len(nl.split("\t")))
    wf.write(nl + "\n")
    wf.write(printTree(root) + "\n")
print(maxlen)
