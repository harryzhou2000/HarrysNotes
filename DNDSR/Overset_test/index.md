---
title: Overset Test
date: 2025-04-11T09:37:51+08:00
type: post
# image: https://harryzhou2000.github.io/resources-0/fractalInf.png
---

目前的Overset方案：均匀笛卡尔背景网格空间索引（HCBG-Overset）

## HCBG-Overset 基本过程

- 准备：
  - N个part，1个BG
  - 根据part的AABB决定BG的范围，并空间剖分
    - 确定h
    - 逻辑划分网格：x-y-z轴
    - 张量积形式剖分
      - 因式分解与剖分问题，采用近似算法
- 距离场计算
  - 插值每个part单元内BG点
    - *不平衡问题*
  - 将part上的边界单元发送到相交的BG单元上
    - 附加此BG单元相交的part体积单元
  - 有边界单元的BG单元可能存在未初始化的BG点
    - 如果边界是壁面，设为-100
    - 否则是+1e300
  - 对剩余未初始化的BG点
    - 通过图连通扩展-100区域，其余为+1e300区域
- 距离场查询
  - 对每个part i，查询part j的距离
    - 在不包含边界的BG单元内，插值
    - 在包含边界单元的BG单元内，精确计算
      - 检测是否在part i流体域内，不在则为1e300/-100
- 挖洞
  - 若 part i 自身的距离小于其他part的距离，此点挖洞
  - 一个单元全部点（目前查询全部节点）挖洞，则为挖洞单元
