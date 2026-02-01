---
title: Ubuntu美化一下
date: 2026-01-03
category: Linux
tags:
  - 美化
description: 每次安装Ubuntu，把它的界面变好看一些
image: https://img.yumeko.site/file/blog/Result.png
status: public
---

# 如何将文件夹中文变成英文
1. 在安装的时候，选择中文，后续不需要安装输入法了，更方便
2. 在设置中**系统**，**区域与语言**，下载英文
![ChangLanguage.png](https://img.yumeko.site/file/articles/UbuntuBeautification/ChangLanguage.png)
3. 重新登录之后，选择**更改文件夹名字**
> [!ATTENTION] 注意
> 注意**不要**勾选**下此不再提醒**
4. 重新设置回中文，重新登陆
> [!ATTENTION] 注意
> 这次要选择不更改文件夹名字

5. 这次可以选择下次不提醒了
# 安装Chrome

```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

sudo dpkg -i google-chrome-stable_current_amd64.deb
```
# 安装Clash

```bash
wget https://github.com/clash-verge-rev/clash-verge-rev/releases/download/v2.4.2/Clash.Verge_2.4.2_amd64.deb
# 可以使用https://kkgithub.com/
sudo dpkg -i Clash.Verge_2.4.2_amd64.deb
```
# 安装字体

安装JetbrainMono字体 [Jetbrains Mono下载](https://www.jetbrains.com/lp/mono/)

```bash
mkdir fonts && cd fonts
wget https://download.jetbrains.com/fonts/JetBrainsMono-2.304.zip
unzip JetBrainsMono-2.304.zip 
sudo cp -r fonts/ttf /usr/local/share/fonts/
sudo fc-cache -fv
```
# 安装Gnome

```
sudo apt install gnome-tweaks
sudo apt install gnome-shell-extensions
```
## 安装插件
在这个[插件网站](https://extensions.gnome.org/)中下载一些gnome的插件
### JustPerfection
![JustPerfection1.png](https://img.yumeko.site/file/articles/UbuntuBeautification/JustPerfection1.png)![JustPerfection2.png](https://img.yumeko.site/file/articles/UbuntuBeautification/JustPerfection2.png)
![JustPerfection3.png](https://img.yumeko.site/file/articles/UbuntuBeautification/JustPerfection3.png)
![JustPerfection4.png](https://img.yumeko.site/file/articles/UbuntuBeautification/JustPerfection4.png)
![JustPerfection5.png](https://img.yumeko.site/file/articles/UbuntuBeautification/JustPerfection5.png)
### Blur my shell
不设置
### Vitals
设置放在面板左侧
### removable-drive-menu
不设置
### Burn My Windows
设置为**Focus**
### Dash to Dock
![Dock1.png](https://img.yumeko.site/file/articles/UbuntuBeautification/Dock1.png)
![Dock2.png](https://img.yumeko.site/file/articles/UbuntuBeautification/Dock2.png)
### UserTheme
见下方安装主题
## 安装主题
### 进入网址找到喜欢的主题

[主题网站](https://www.gnome-look.org/browse?cat=135&ord=latest)
1. 找到一个主题![theme1.png](https://img.yumeko.site/file/articles/UbuntuBeautification/theme1.png)
2. 然后点击他的Download，进入他的github主页![theme2.png](https://img.yumeko.site/file/articles/UbuntuBeautification/theme2.png)![theme3.png](https://img.yumeko.site/file/articles/UbuntuBeautification/theme3.png)
3. 进入**releases**页面，下载源码![theme4.png](https://img.yumeko.site/file/articles/UbuntuBeautification/theme4.png)
### 解压主题

```bash
# 如果是gz结尾
sudo tar zxvf {ThemeName}.tar.gz
# 如果是xz结尾
sudo tar xvf {ThemeName}.tar.xz

mkdir ~/gnome-themes
sudo mv {ThemeName}/ ~/gnome-themes

cd ~/gnome-themes/{ThemeName}/
sudo ./install.sh
```
### 在UserThemes插件中设置主题即可
## 使用动态壁纸
这是[Hidamari](https://github.com/jeffshee/hidamari)的github源码, 右侧About中有下载链接
![Hidamari源码](https://nyeimg.asia/img/articles/Ubuntu美化/Hidamari源码.png)

1. 在[Flathub](https://flathub.org/en/apps/io.github.jeffshee.Hidamari)中下载Hidamari
2. 如果没有安装Flathub, 请在[Flathub安装网站](https://flathub.org/en/setup)中下载Flathub, 然后回到上一步下载Hidamari
3. 在[视频壁纸网站](https://motionbgs.com/)中下载mp4格式的动态壁纸
4. 下载在一个文件夹，例如: wallpapaer中
5. 在Hidamari应用中选择wallpaper为动态壁纸文件夹, 然后刷新即可
6. 在应用右上角选择开机自动启动
![Hidamari应用](https://nyeimg.asia/img/articles/Ubuntu美化/Hidamari应用.png)
## 使用conky

1. 下载conky管理软件
```
sudo apt install conky-all conky-manager
```
2. 安装conky插件
在[conky插件网站](https://www.pling.com/browse?cat=124&ord=latest)中下载喜欢的conky文件, 解压之后放到~/.config/conky/文件夹下, 例如这是Maia插件的界面
![conky文件夹](https://nyeimg.asia/img/articles/Ubuntu美化/conky文件夹.png)
3. 字体下载
一般conky文件中会放fonts文件夹,
* 将fonts的ttf都放到/usr/share/fonts中
* 在conky插件的文件夹下
```
sudo mkdir /usr/share/fonts/conky
sudo mv fonts/* /usr/share/fonts/conky
```
* 更新字体
```
fc-cache -fv
```
4. 使用conky插件即可

在插件文件夹中
```
./start.sh
```
5. 常见的bug
* 插件背景是黑色
修改`own_window_argb_visual = false`即可
![ConkyConfig](https://img.yumeko.site/file/articles/UbuntuBeautification/ConkyConfig.png)
* 插件中出现了乱码(conky用的是lua语言)
例如这是一段config中的源代码
```
conky.text = [[
${execi 200 ~/.config/conky/Maia/scripts/weather-v3.0.sh -g}\
${alignr 320}${font Comfortaa:size=120}${time %d}${font}
${alignr 327}${voffset 20}${font Comfortaa:size=20}${time %B %Y}
${alignr 327}${voffset -3}${font Comfortaa:size=13}Today's ${time %A}${font}
${offset 310}${voffset -210}${font feather::size=50}${execi 15 ~/.config/conky/Maia/scripts/weather-text-icon}${goto 410}${font Comfortaa:size=45}${execi 100 ~/.config/conky/Maia/scripts/weather-v3.0.sh -t}${voffset -23}${font Comfortaa:size=25}°
${offset 310}${voffset 20}${font Comfortaa:size=11}${execi 100 ~/.config/conky/Maia/scripts/weather-v3.0.sh -n}
${offset 310}${voffset 5}${font Comfortaa:bold:size=15}${execi 100 ~/.config/conky/Maia/scripts/weather-v3.0.sh -m}
${offset 310}${voffset 1}${font Comfortaa:size=11}Feels like ${execi 100 ~/.config/conky/Maia/scripts/weather-v3.0.sh -f}°F
${offset 310}${voffset 21}${font Comfortaa:size=29}${time %I:%M%^p}
]]
```
*  可能是字体有问题
    * 例如这个是conky插件的某一个config,font后面的`Comfortaa`就是一个字体，如果这一行出现了方块乱码, 将`Comfortaa`调整为`Noto Sans`(这是linux自带的中文字体)
    * 如果这个乱码变成了中文, 就可以按下面的更改, 把时区改了
```
# 原代码，出现了乱码
${alignr 327}${voffset -3}${font Comfortaa:size=13}Today's ${time %A}${font}
```
```
${alignr 327}${voffset -3}${font Comfortaa:size=13}${execi 1 LC_TIME=en_US.UTF-8 date +%A}
```

* 可能是代码本身错误
    * 在lua语言中, 年月日用的是`%Y/%m/%d`, 所以这里不能用`%B`
    * 这里要改成`%m`
```
${alignr 327}${voffset 20}${font Comfortaa:size=20}${time %B %Y}
```
```
${alignr 327}${voffset 20}${font Comfortaa:size=20}${time %m %Y}
```
* 如果你的conky插件中有CPU、RAM等设备检测，要把你的设备名字改成你的设备名字

* 如果你的conky中有天气显示的插件, 在scripts文件中查看作者的说明, 更改city的id
**其他更详细的错误，都可以丢给ai，问问ai**
# grub引导主题

1. 下载 GRUB 主题
首先，你需要从[GRUB网站](https://www.gnome-look.org/browse?cat=109&ord=latest)下载一个GRUB主题

2. 解压缩主题
下载的主题通常会以压缩文件格式（.zip 或 .tar.gz）提供。需要解压缩它。

例如，如果你的主题文件名为 grub-theme.zip，可以使用以下命令解压：
```
unzip grub-theme.zip
```
如果是 .tar.gz 文件：
```
tar -xzvf grub-theme.tar.gz
```
3. 移动主题文件
* 如果你的GRUB中有`install.sh`, 那你直接启动就可以转移到下一步了
* 将解压后的主题文件夹移动到 GRUB 主题目录。通常，GRUB 主题位于 /boot/grub/themes/ 目录下。如果没有这个目录，可以创建它。

```
sudo mkdir -p /boot/grub/themes 
sudo mv /path/to/your/theme-folder /boot/grub/themes/
```
确保替换 `/path/to/your/theme-folder` 为你的实际路径。

4. 配置 GRUB 主题

* 在 GRUB 配置文件中设置新主题。编辑 `/etc/default/grub` 文件：
```
sudo vim /etc/default/grub
```
* 找到 GRUB_THEME 这一行，如果没有这个行，可以自行添加一行。例如：
```
GRUB_THEME="/boot/grub/themes/theme-folder/theme.txt"
```
替换 theme-folder 和 theme.txt 为你主题文件夹中的实际名称和文件名。

5. 更新 GRUB
完成设置后，你需要更新 GRUB(很重要)，以便加载新的主题：
```
sudo update-grub
```
6. 重启计算机
最后重启计算机查看新的 GRUB 主题：
```
sudo reboot
```
7. 设置GRUB字体大小

如果你觉得GRUB分辨率太小了, 可以在`/etc/default/grub`中编辑, 注意!乘法就是一个`x`字母, 不是`*`符号
```
GRUB_GFXMODE=1024x768
```
8. 设置启动项目顺序

在ubuntu中，所有的启动项都在/etc/grub.d/文件夹中
例如10_linux就是我们的ubuntu系统, 启动项顺序就是文件夹的顺序, 所以10_linux(ubuntu)就在30_os-prober(windows)前面
```
sudo mv 30_os-prober 10_os-prober
```
这样就可以把windows的启动项提前到ubuntu背后了, 同理也可以把windows提前到ubuntu前面

原因: 查看`/boot/grub/grub.cfg`文件, 在文件下半部分就是grub的启动项信息(grub自动更新生成), 其他内容清自行查阅资料(如子启动项，如何删除启动项), 可以在[ArchWiki](https://wiki.archlinux.org/title/Main_page)中学习
9. 更新GRUB 
完成设置后，你需要更新 GRUB(很重要)，以便加载新的主题：
```
sudo update-grub
```
# 最后效果

![Result](https://img.yumeko.site/file/articles/UbuntuBeautification/Result.png)