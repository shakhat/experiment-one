# Bug: Index operator crashes on Ascend NPU when type int64 

The following code crashes:
```python
dev = "npu"
r = torch.tensor([1], dtype=torch.int64, device=dev)
s = torch.tensor([[0]*252], dtype=torch.int64, device=dev)
r[s]
```

Stacktrace:
```
RuntimeError: InnerRun:torch_npu/csrc/framework/OpParamMaker.cpp:197 NPU error, error code is 500002
[ERROR] 2024-07-03-10:49:44 (PID:910666, Device:0, RankID:-1) ERR01100 OPS call acl api failed
[Error]: A GE error occurs in the system.
        Rectify the fault based on the error information in the ascend log.
E40021: Failed to compile Op [Index2]. (oppath: [Compile /usr/local/Ascend/ascend-toolkit/7.0.0/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/index.py failed with errormsg/stack: File "/usr/local/Ascend/ascend-toolkit/7.0.0/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/index.py", line 536, in 
\t->  x_tail = self.tik_instance.Scalar(init_value=self.input_batch_num_0 % self.batch_align)
==============================================================================
Traceback (most recent call last):
  193: TVMFuncCall
  192: ascend_tvm::runtime::TypedPackedFunc<ascend_tvm::tir::Stmt (ascend_tvm::tir::Stmt, std::string const&, ascend_tvm::runtime::Array<ascend_tvm::PrimExpr, void> const&)>::AssignTypedLambda<ascend_tvm::tir::Stmt (*)(ascend_tvm::tir::Stmt, std::string const&, ascend_tvm::runtime::Array<ascend_tvm::PrimExpr, void> const&)>(ascend_tvm::tir::Stmt (*)(ascend_tvm::tir::Stmt, std::string const&, ascend_tvm::runtime::Array<ascend_tvm::PrimExpr, void> const&), std::string)::{lambda(ascend_tvm::runtime::TVMArgs const&, ascend_tvm::runtime::TVMRetValue*)#1}::operator()(ascend_tvm::runtime::TVMArgs const&, ascend_tvm::runtime::TVMRetValue*) const
  191: ascend_tvm::tir::transform::DynamicCombineStaticPass(ascend_tvm::tir::Stmt, std::string const&, ascend_tvm::runtime::Array<ascend_tvm::PrimExpr, void> const&)
  190: ascend_tvm::tir::ConstantFoldingOnce(ascend_tvm::tir::Stmt, int)
  189: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  188: 0x0000ffff7ad89903
  187: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  186: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  185: 0x0000ffff7ad89903
  184: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  183: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  182: 0x0000ffff7ad89903
  181: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  180: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  179: 0x0000ffff7ad89903
  178: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  177: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  176: 0x0000ffff7ad89903
  175: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  174: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  173: 0x0000ffff7ad89903
  172: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  171: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  170: 0x0000ffff7ad89903
  169: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  168: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  167: 0x0000ffff7ad89903
  166: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  165: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  164: 0x0000ffff7ad89903
  163: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  162: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  161: 0x0000ffff7ad89903
  160: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  159: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  158: 0x0000ffff7ad89903
  157: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  156: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  155: 0x0000ffff7ad89903
  154: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  153: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  152: 0x0000ffff7ad89b8b
  151: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  150: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  149: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  148: 0x0000ffff7ad89b8b
  147: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  146: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  145: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  144: 0x0000ffff7ad89b8b
  143: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  142: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  141: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  140: 0x0000ffff7ad89b8b
  139: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  138: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  137: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  136: 0x0000ffff7ad89b8b
  135: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  134: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  133: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  132: 0x0000ffff7ad89b8b
  131: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  130: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  129: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  128: 0x0000ffff7ad89b8b
  127: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  126: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  125: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  124: 0x0000ffff7ad89b8b
  123: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  122: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  121: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  120: 0x0000ffff7ad89b8b
  119: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  118: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  117: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  116: 0x0000ffff7ad89b8b
  115: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  114: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  113: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  112: 0x0000ffff7ad89b8b
  111: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  110: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  109: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  108: 0x0000ffff7ad89b8b
  107: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  106: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  105: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  104: 0x0000ffff7ad89b8b
  103: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  102: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  101: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  100: 0x0000ffff7ad89b8b
  99: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  98: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  97: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  96: 0x0000ffff7ad89b8b
  95: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  94: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  93: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  92: 0x0000ffff7ad89b8b
  91: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  90: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  89: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  88: 0x0000ffff7ad89b8b
  87: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  86: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  85: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  84: 0x0000ffff7ad89b8b
  83: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  82: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  81: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  80: 0x0000ffff7ad89b8b
  79: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  78: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  77: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  76: 0x0000ffff7ad89b8b
  75: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  74: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  73: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  72: 0x0000ffff7ad879b3
  71: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  70: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  69: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  68: 0x0000ffff7ad89b8b
  67: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  66: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  65: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  64: 0x0000ffff7ad89b8b
  63: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  62: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  61: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  60: 0x0000ffff7ad879b3
  59: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  58: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  57: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  56: 0x0000ffff7ad89903
  55: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  54: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  53: 0x0000ffff7ad8817b
  52: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::IfThenElseNode const*)
  51: ascend_tvm::tir::ConstantFold::MutateIfCondConst(ascend_tvm::tir::IfThenElseNode const&, ascend_tvm::PrimExpr)
  50: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  49: 0x0000ffff7ad8817b
  48: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::IfThenElseNode const*)
  47: ascend_tvm::tir::ConstantFold::MutateIfCondConst(ascend_tvm::tir::IfThenElseNode const&, ascend_tvm::PrimExpr)
  46: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  45: 0x0000ffff7ad879b3
  44: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  43: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  42: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  41: 0x0000ffff7ad84dbb
  40: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::ForNode const*)
  39: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::ForNode const*)
  38: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  37: 0x0000ffff7ad89903
  36: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AttrStmtNode const*)
  35: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  34: 0x0000ffff7ad89b8b
  33: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  32: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  31: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  30: 0x0000ffff7ad879b3
  29: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  28: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  27: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  26: 0x0000ffff7ad89b8b
  25: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  24: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  23: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  22: 0x0000ffff7ad879b3
  21: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  20: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  19: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  18: 0x0000ffff7ad89b8b
  17: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  16: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::AllocateNode const*)
  15: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  14: 0x0000ffff7ad879b3
  13: ascend_tvm::tir::StmtMutator::VisitStmt_(ascend_tvm::tir::SeqStmtNode const*)
  12: void ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void>::MutateByApply<ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1}>(ascend_tvm::tir::StmtMutator::Internal::Mutate(ascend_tvm::tir::StmtMutator*, ascend_tvm::runtime::Array<ascend_tvm::tir::Stmt, void> const&)::{lambda(ascend_tvm::tir::Stmt const&)#1})
  11: ascend_tvm::tir::StmtMutator::VisitStmt(ascend_tvm::tir::Stmt const&)
  10: 0x0000ffff7ad8910b
  9: ascend_tvm::tir::ConstantFold::VisitStmt_(ascend_tvm::tir::StoreNode const*)
  8: ascend_tvm::tir::Simplify(ascend_tvm::PrimExpr, ascend_tvm::runtime::Map<ascend_tvm::tir::Var, ascend_tvm::Range, void, void>)
  7: ascend_tvm::arith::Analyzer::Simplify(ascend_tvm::PrimExpr const&, int)
  6: ascend_tvm::arith::RewriteSimplifier::operator()(ascend_tvm::PrimExpr const&)
  5: non-virtual thunk to ascend_tvm::tir::StmtExprMutator::VisitExpr(ascend_tvm::PrimExpr const&)
  4: _ZZN10ascend_tvm3tir11ExprFunctorIFNS_8PrimExprERKS2_EE10InitVTableEvENUlRKNS_7runtime9ObjectRef
  3: ascend_tvm::arith::RewriteSimplifier::Impl::VisitExpr_(ascend_tvm::tir::ModNode const*)
  2: ascend_tvm::PrimExpr ascend_tvm::arith::TryConstFold<ascend_tvm::tir::Mod>(ascend_tvm::PrimExpr, ascend_tvm::PrimExpr)
  1: ascend_tvm::runtime::detail::LogFatal::Entry::Finalize()
  0: ascend_tvm::runtime::Backtrace()
  File "canonical_simplify.cc", line 329

TVMError: [EB9000] ---------------------------------------------------------------An error occurred during the execution of TVM.For more information, please see: https://tvm.apache.org/docs/errors.html---------------------------------------------------------------  Check failed: pb->value != 0 (0 vs. 0) : Divide by zero (int64)1%(int64)0', 'errPcause': 'N/A', 'errSolution': 'N/A'}
], optype: [Index])
        Solution: See the host log for details, and then check the Python stack where the error log is reported.
        TraceBack (most recent call last):
        Compile op[Index2] failed, oppath[/usr/local/Ascend/ascend-toolkit/7.0.0/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/index.py], optype[Index], taskID[2]. Please check op's compilation error message.[FUNC:ReportBuildErrMessage][FILE:fusion_manager.cc][LINE:771]
        [SubGraphOpt][Compile][ProcFailedCompTask] Thread[281472368538080] recompile single op[Index2] failed[FUNC:ProcessAllFailedCompileTasks][FILE:tbe_op_store_adapter.cc][LINE:954]
        [SubGraphOpt][Compile][ParalCompOp] Thread[281472368538080] process fail task failed[FUNC:ParallelCompileOp][FILE:tbe_op_store_adapter.cc][LINE:1001]
        [SubGraphOpt][Compile][CompOpOnly] CompileOp failed.[FUNC:CompileOpOnly][FILE:op_compiler.cc][LINE:1127]
        [GraphOpt][FusedGraph][RunCompile] Failed to compile graph with compiler Normal mode Op Compiler[FUNC:SubGraphCompile][FILE:fe_graph_optimizer.cc][LINE:1292]
        Call OptimizeFusedGraph failed, ret:-1, engine_name:AIcoreEngine, graph_name:partition0_rank1_new_sub_graph4[FUNC:OptimizeSubGraph][FILE:graph_optimize.cc][LINE:131]
        subgraph 0 optimize failed[FUNC:OptimizeSubGraphWithMultiThreads][FILE:graph_manager.cc][LINE:996]
        build graph failed, graph id:1, ret:-1[FUNC:BuildModelWithGraphId][FILE:ge_generator.cc][LINE:1615]
        [Build][SingleOpModel]call ge interface generator.BuildSingleOpModel failed. ge result = 4294967295[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
        [Build][Op]Fail to build op model[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
        build op model failed, result = 500002[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
```

With a different size of tensor `s` the code works, but the result is wrong and it execution leads to memory corruption.
