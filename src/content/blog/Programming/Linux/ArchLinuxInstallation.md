---
title: 安装Arch
date: 2026-07-04
category: 编程/Linux
tags:
  - Linux
  - Arch
  - 美化
  - 安装
description: 安装一个Arch系统
image: https://img.yumeko.site/file/blog/cover/enjoy-archlinux.webp
status: published
---
# 在U盘中下载iso镜像

## 下载ISO镜像

在[清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/archlinux/iso/latest/)中下载最新的iso镜像

![image-20260706095634470](https://img.yumeko.site/file/blog/articles/image-20260706095634470.webp)

## U盘烧录iso镜像

具体过程参考[[USBBooting|制作启动盘]]这篇文章，Rufus和Ventoy任选其一即可，推荐Ventoy

如果你希望U盘不用烧录的情况下，使用iso镜像，请参考如何让U盘使用多个iso镜像这篇文章，使用Ventoy项目，将U盘变成多个iso镜像和其他文件夹共存状态。

# 准备工作

## 准备一个空闲磁盘

使用[傲梅助手](https://www.disktool.cn/download.html)，在Windows中开辟一个空闲磁盘，然后提交，这个空的地方就是用来安装Archlinux

![image-20260706100205812](https://img.yumeko.site/file/blog/articles/image-20260706100205812.webp)



## 重启，使用USB启动

插入U盘，在BIOS中选择USB启动，会进入安装界面

### BIOS引导安装界面

![image-20260706100741642](https://img.yumeko.site/file/blog/articles/image-20260706100741642.webp)

### UEFI引导启动界面

![image-20260706113259353](https://img.yumeko.site/file/blog/articles/image-20260706113259353.webp)

两者区别主要在grub安装部分，稍后会说明

# 开始安装

## 连接网络

### 有线网络

如果你是有线网络，应该是不会出问题的，直接测试网络即可

### 无线网络

使用`iwctl`联网

```bash
# 列出联网设备
iwctl device list
# 扫描网络
iwctl station {设备名字} scan
# 获取网络
iwctl station {设备名字} get-networks
# 连接网络
iwctl station {设备名字} connect {wifi名字}
```

### 测试网络

```bash
ping baidu.com
```

![image-20260706101419021](https://img.yumeko.site/file/blog/articles/image-20260706101419021.webp)

## 分盘

### 查看盘块情况

```bash
# 树形结构
lsblk -f 
# 列表形式
lsblk -l
```

![image-20260706101546170](https://img.yumeko.site/file/blog/articles/image-20260706101546170.webp)

![image-20260706101643658](https://img.yumeko.site/file/blog/articles/image-20260706101643658.webp)

这里可以看到`sda`盘中是空挂载，而且`size`为`20G`，所以这个就是我们的盘

### 开始分盘

```
cfdisk /dev/{空闲盘名字}
```

> [!ATTENTION] 空闲盘怎么填
> 如果你是树形结构，比如
> ```
> sda
|_sda1 # win C
|_sda2 # win D
|_sda3 # win E
|_sda4 # 空闲盘
> ```
> 如果你想使用sda4作为linux的盘，那你的空闲盘应该是`/dev/sda`
> ![image-20260706102021130](https://img.yumeko.site/file/blog/articles/image-20260706102021130.webp)

### 分EFI启动项盘

#### 如果是UEFI引导

EFI分区给500MB就够了

![image-20260706103233222](https://img.yumeko.site/file/blog/articles/image-20260706103233222.webp)

> [!ATTENTION] 设置完成后
> 一定要选择**Write**，一定要写**yes**！！！

#### 如果是BIOS引导

直接跳过EFI分区

### 分swap交换分区

回车`Free Space`，输入`Swap`分区的大小，然后选择`Type`

![image-20260706103351070](https://img.yumeko.site/file/blog/articles/image-20260706103351070.webp)

![image-20260706102216507](https://img.yumeko.site/file/blog/articles/image-20260706102216507.webp)

> [!ATTENTION] 设置完成后
> 一定要选择**Write**，一定要写**yes**！！！

### 分文件根目录

直接`New`就是默认的`Linux filesystem`

> [!ATTENTION] 设置完成后
> 一定要选择**Write**，一定要写**yes**！！！

### 检查磁盘情况

![image-20260706102704177](https://img.yumeko.site/file/blog/articles/image-20260706102704177.webp)

## 格式化分盘

### 格式化文件根目录

```bash
mkfs.ext4 /dev/{Linux Filesystem区名字}
```

> [!ATTENTION] 格式化注意
> `mkfs.ext4`会清空盘数据

### 格式化swap交换分区

```bash
mkswap /dev/{swap分区}
```
### EFI分区

> [!ATTENTION] 双系统注意
> 如果你有Windows系统或者其他系统，这个EFI就不用格式化了，因为会清空你的EFI启动项，直接跳过这一步就可以

```bash
mkfs.fat -F 32 /dev/{EFI 分区盘}
```

### 验证是否格式化

使用`lsblk -f`指令可以看到盘目前的`FSTYPE`

![image-20260706103841352](https://img.yumeko.site/file/blog/articles/image-20260706103841352.webp)



## 挂载分区

### 挂载根目录

```bash
mount /dev/{file system盘} /mnt
```

### 挂载swap交换分区

```bash
swapon /dev/{swap分区盘}
```

### 挂载EFI分区

```bash
mount --mkdir /dev/{EFI分区} /mnt/boot
```

## 安装基本的系统

### 更新镜像源

```bash
vim /etc/pacman.d/mirrorlist
```

可以把`China`的镜像源提到前面，关于vim的操作可以看新文章

或者你可以一键到位

```bash
sudo sed -i 's/^/#/' /etc/pacman.d/mirrorlist
```

这样可以把所有的镜像源全部注释，然后定位到`China`的源，把想要的给取消注释就行了

### 安装Linux基本系统

```bash
pacstrap -K /mnt base linux linux-firmware base-devel
```

安装好之后，这个linux系统就是基本安装好了，但是还有很多东西没有，比如**启动项**，**EFI管理**，**网络管理**

# 进入系统

## 载入系统

### 将挂载信息载入系统

```bash
genfstab -U /mnt >> /mnt/etc/fstab
```

### 进入arch-chroot模式

```bash
arch-chroot /mnt
```

## 对齐时钟

### 设置时区

```bash
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```

### 对准时钟

```bash
hwclock --systohc
```

### 验证时间

```bash
date
```

![image-20260706114701603](https://img.yumeko.site/file/blog/articles/image-20260706114701603.webp)

## 设置语言

> [!ATTENTION] 为什么不用中文
> 设置为中文会出现乱码，所以暂时设置为英文，进入桌面端可以改

### 取消英文语言注释

```bash
vim /etc/locale.gen
```

![image-20260706110522083](https://img.yumeko.site/file/blog/articles/image-20260706110522083.webp)

取消英文UTF-8的注释即可

### 加载语言

```bash
locale-gen
```

![image-20260706110628575](https://img.yumeko.site/file/blog/articles/image-20260706110628575.webp)

### 设置系统语言

```bash
vim /etc/locale.conf
```

插入语言内容

```bash
LANG=en_US.UTF-8
```



![image-20260706110727795](https://img.yumeko.site/file/blog/articles/image-20260706110727795.webp)

## 设置用户权限

### 管理员

在`root`账户下，使用下面的指令设置`root`密码

```bash
passwd
```

### 普通用户

在`root`账户下，使用下面的指令创建一个用户

```bash
useradd -m {用户名字}
```

给用户创建一个密码

```bash
passwd {用户名字}
```

### 普通用户使用sudo

编辑`/etc/sudoers`文件，让普通用户可以使用`sudo`命令

```bash
vim /etc/sudoers
```

在root用户下方加上普通用户的权限

``` 
{用户名字} ALL=(ALL) ALL
```

![image-20260706111741958](https://img.yumeko.site/file/blog/articles/image-20260706111741958.webp)

### 编辑一个电脑名字

在终端中会显示`{用户名字}@{hostname}`，这里的`hostname`是可编辑

```
vim /etc/hostname
```

取一个自己喜欢的名字，然后保存

## 下载一些管理程序

```bash
pacman -S grub efibootmgr os-prober
```

### 编辑grub

```bash
vim /etc/default/grub
```

为了双系统可以搜索到其他的操作系统，在grub设置中必须取消最后一行

![image-20260706112322504](https://img.yumeko.site/file/blog/articles/image-20260706112322504.webp)

### 安装grub

要回到根目录

```bash
cd /
```

#### 如果是UEFI引导

```
grub-install --target=x86_64-efi --efi-directory=boot --bootloader-id=GRUB
```

![image-20260706114217562](https://img.yumeko.site/file/blog/articles/image-20260706114217562.webp)

#### 如果是BIOS引导

```
grub-install --target=i386-pc /dev/{系统安装的硬盘}
```

> [!ATTENTION] 注意
> 这里是硬盘，不是分区



### 生成grub配置文件

```
grub-mkconfig -o /boot/grub/grub.cfg
```

##  设置网络

### 安装网络包

```bash
pacman -S networkmanager iwd
```

### 开机自启动

注意大写

```
systemctl enable NetworkManager
```

## 退出系统，重启

```bash
exit
```

```bash
reboot
```

# 重启后验证

![image-20260706120855324](https://img.yumeko.site/file/blog/articles/image-20260706120855324.webp)

接下来可以看arch的其他内容了
