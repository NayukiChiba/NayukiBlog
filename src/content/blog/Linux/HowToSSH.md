---
title: 对于远程控制Linux出现的一些问题
date: 2026-01-26
category: Linux
tags:
  - 服务器
  - Bug
description: 收集链接服务器时候出现的各种问题
image: https://img.yumeko.site/file/articles/HowToSSH/HowToSSH.png
status: public
---
# 正确的SSH连接过程
* 在控制台中输入
```bash
ssh {username}@{ip}
```
* 对于是否保留指纹，可以填yes
* 然后输入密码，就可以进入服务器中了
# 出现WARNING警告
如果出现下列警告
```bash
$ ssh ec2-user@ec2-192-168-1-1.compute-1.amazonaws.com
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ECDSA key sent by the remote host is
SHA256:hotsxb/qVi1/ycUU2wXF6mfGH++Yk7WYZv0r+tIhg4I.
Please contact your system administrator.
Add correct host key in /Users/scott/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /Users/scott/.ssh/known_hosts:47
ECDSA host key for ec2-192-168-1-1.compute-1.amazonaws.com has changed and you have requested strict checking.
Host key verification failed.
```
可以在控制台中输入：
```bash
ssh-keygen -R {ip}
```
然后重新ssh连接服务器即可