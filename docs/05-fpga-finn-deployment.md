# FPGA 部署：FINN Full Build 与板端验证

**这条路线没有走完。** bitstream 生成成功、板端能加载、加速器能执行，但输出与
golden 不一致。这里如实记录走到了哪里、卡在哪里。

一个能把失败边界精确到具体环节的项目，比一个声称全部成功的项目更可信。

## 6.1 Windows 原生 FINN 路线的打通

FINN 官方建议在 Linux + Docker 下运行，而本项目主力构建机是 Windows。需要的适配：

1. **双环境分离**：`fashion-frontier-cu`（训练/量化）与 `fashion-frontier-finn-win`（FINN 构建）独立
   管理，避免环境污染
2. **Windows-aware 工具发现**：适配 AMD Unified 2025.2 的 `vitis-run --mode hls`
   调用方式
3. **路径规范化**：Tcl 路径、构建目录路径做 Windows 兼容处理
4. **Bug 修复**：
   - `create_stitched_ip.py`：Windows 下 stitched-IP 实例名过长
   - `templates.py`：Vitis HLS 2022.2 的 include 路径规范化
   - `replace_verilog_relpaths.py`：Windows 下 `$readmemh` 反斜杠路径错误

### 一个"失败即里程碑"的时刻

第一次 full build 失败在 placement 阶段：**需要 12 312 slices，可用 11 448**。

这本身是重要里程碑 —— 它证明 Windows 原生构建路线**本质可行**，把问题从"环境是否
可用"变成了"设计能否装下"。这两类问题的排查成本差一个量级。

## 6.2 资源收敛：S16 Folding

采用更保守的 folding 配置（`folding_config.resource_s16.json`），通过**降低 SIMD
折叠数**实现资源收缩，成功生成了第一个完整 bitstream 和 deploy package
（`build_full_fit_s16_r2`）。

## 6.3 板端第一个坑不是数值，是缓存

板端部署的首个关键问题不是数值错误，而是 **overlay metadata 缓存复用**：系统把自
定义 overlay 解析成了 base overlay，原因是

```
/home/xilinx/pynq/pl_server/_current_metadata.pkl
```

保留了旧缓存。清除后，板端才首次正确识别 `idma0`、`odma0`、`zynq_ps`，部署工作流
才真正进入数值验证阶段。

> 经验：**"硬件没反应"和"硬件算错了"是完全不同的故障层。** 先确认 overlay 真的是
> 你新烧的那个。

## 6.4 分层验证：把问题定位到正确的层 ★

overlay 正确加载后，板端加速器可以执行，但输出与 golden 不一致（预测 label=8，期望
label=9）。没有去猜，而是做了一层一层的验证：

| 验证层次 | 验证方式 | 状态 |
|---|---|---|
| PyTorch 训练逻辑 | 单元测试 | **Golden** |
| ONNX/QONNX 导出 | 主机端推理比对 | **Golden** |
| FINN cppsim（全模型） | C++ 仿真 | **Golden** |
| FINN rtlsim（单节点） | RTL 仿真 | **Golden** |
| Stitched-IP rtlsim | 集成 RTL 仿真 | **Non-golden**（输出 3，期望 9） |
| Board execution | 真实板端 | **Stable but wrong**（输出 8，期望 9） |

结论非常干净：**训练没问题、导出没问题、单节点 RTL 没问题，问题精确落在 stitched
集成路径。**

这是整个项目里最值得留下的一张表。没有它，"板端输出错了"这句话可以指向六个不同的
地方，每一个的排查成本都是几天。有了它，问题范围从"整个项目"缩小到"一个集成环节"。

> **分层验证是硬件调试的基本原则。** 任何问题都必须被定位到正确的层，不能靠单一
> 实验结论跨层下判断。

顺带一条经验：**如果一个"应该更快"的部署路径反而更慢了，先怀疑链路，不要怀疑
模型。** 数量级上的反常几乎从来没有模型层面的解释。

## 6.5 SIMD 契约不一致

后续分析发现，某些大 MVAU 节点存在 **ONNX 图属性与 HLS 代码生成结果不一致**：

- 图中记录 `SIMD = 1`
- 但 HLS 因 $\lceil MW/1024 \rceil$ 约束，实际生成了 `SIMD = 2`

这是纯粹的**工具链内部语义契约问题** —— 不是模型问题，也不是用户配置问题，而是图
里的声明和代码生成器实际行为对不上。它解释了此前 late-path 区域的诸多"神秘"异常。

## 6.6 最终状态与工程边界

项目结束时，这条线的状态：

- ✅ Windows 原生 full build 已完整打通，多个 bitstream / hwh / driver / deploy
  包已成功生成
- ✅ 板端 metadata / overlay / driver / hardware-semantic validation 体系已建立
- ⚠️ BRAM fit：重训练候选需要 RAMB18/RAMB36 共 **284 个，可用 280 个**（差距极小）
- ❌ **最终数值正确性闭环未完成**，问题边界已收缩至 stitched-IP RTL 集成路径

差 4 个 BRAM 这件事很折磨人，但它也说明设计已经贴着资源上限了 —— 再优化一点就能
装下。

## 如果要继续推进

1. **解决 BRAM fit**：热点在 `memstream_axi_wrapper`，可尝试调整 `ram_style`
   （LUTRAM vs BRAM）或减少 weight streaming 并发度
2. **解决 stitched-IP 正确性**：在受信任 Linux 环境下重跑 stitched rtlsim，排除
   Windows 工具链残留影响
3. **出发点**：`finnconv_direct_head8_kd_noncheat`（已完成
   estimate / codegen / ipgen / OOC synthesis）
4. **预期收益**：若数据流化完全正确，4W4A 模型理论吞吐可达 **1000–10000 img/s**
   量级，远超 CPU 路线的 445 img/s

## 下一步

[06 - BNN 路线](./06-bnn-route.md)
