# A4 V06 真实考官证明材料（策略组 2026-11-14 自 Studio 直取）

- `opp_master2r.py` — md5 `2aebfbc1008d3475ece8e178696e292d`
  源：Studio `/Users/arm/junyang/carbon/code/exp/opp_master2r.py`（策略组一手直读/直测；
  DGX 有同字节副本，作战室/Studio 双方回执互证）。exam-only 工件，**不在仓内**（其
  docstring 自述：包装 Master2 公开源码、never in a training pool or a submission）。
- `master2_src.py` — md5 `404114a0b5e422516322f62e9ddb6870`
  源：Studio 同目录。它是 wrapper 以 spec_from_file_location 相对 `__file__` 加载的
  第一个候选路径（同目录优先）；第二候选 `../../data/public_src/master2.py` 在同目录
  副本存在时不被使用（与生产解析顺序一致）。本目录两件并置即生产布局。

证明口径（策略组条件）：新版 cnet2_train 以真实 opp_master2r.py 完成选模开跑设置，
`selection_examiner_deps` 须含 master2_src.py（md5 404114a0…），考官身份保持
2aebfbc1 不变，短 fixture 选模跑通。
