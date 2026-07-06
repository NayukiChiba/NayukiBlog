---
title: 将USB作为系统启动盘
date: 2026-07-06
category: 编程/Linux
tags:
  - Linux
  - 安装
description: 介绍使用Rufus和Ventoy两种烧录U盘的方法
image: https://img.yumeko.site/file/blog/cover/Ventoy_Logo.webp
status: published
---
# 阅读前须知
所有的启动盘，只要是用U盘，就一定会格式化，但是二者有区别
- Ventoy格式化一次之后，以后iso镜像与其他文件共存，正常使用
- Rufus只能写一个系统，而且不可再写入其他内容，除非再次格式化成正常U盘
# 使用Ventoy（推荐）

Ventoy是一款开源的多系统启动U盘制作工具，无需反复格式化U盘，直接复制ISO等镜像文件即可启动。

在[官网](https://www.ventoy.net/cn/index.html)或者[github主页](https://github.com/ventoy/Ventoy)中下载Ventoy，然后安装

![image-20260706191545157](D:/YeYann/Img/blog/articles/image-20260706191545157.png)

启动Ventoy后，插入U盘就会在设备中看见U盘，然后安装即可

# 使用Rufus

Rufus是一款免费开源、体积小巧的USB启动盘制作工具。

在[官网](https://rufus.ie/zh/)中下载Rufus

Rufus会格式化U盘所有内容，完全变成一个启动盘，**无法**往里面**写入**任何数据，只能用于**一个系统**的安装

![image-20260706192012240](D:/YeYann/Img/blog/articles/image-20260706192012240.png)

选择设备和镜像文件之后，选择分区类型

## 查看分区类型

打开磁盘管理，然后对磁盘**右键**选择**属性**，选择**卷**，可以看到磁盘分区形式为**GPT**，还有一种是**MBR**，按需选择

![image-20260706192307363](D:/YeYann/Img/blog/articles/image-20260706192307363.png)

配置好了之后可以使用Rufus烧录U盘了
# 恢复成正常U盘
Ventoy不用恢复，本来就是正常的

Rufus启动盘需要使用DiskGenius或者其他的分区操作软件，把U盘变成空闲分区，然后再格式化成新分区，就可以再次正常使用了