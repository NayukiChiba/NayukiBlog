---
title: 如何用服务器直接搭建图床
date: 2026-01-01
category: 小巧思
tags:
  - 免费
  - 服务器
  - 有趣
  - 资源
description: 没有大米，也不想搞复杂配置怎么办，用sftp跟博客一起做图床
image: https://img.yumeko.site/file/blog/SFTPImgbedDeployment.jpg
status: published
---
## 如何用服务器直接搭建图床
### 服务器端

1. 创建图片文件夹img 

![服务器img文件夹](https://img.yumeko.site/file/articles/SFTPImgbedDeployment/imgFolder.png)

2. 设置nginx.conf

* 目标：当域名后面有img的时候，映射到html/img下

* 在server中加上location的地址映射

```nginx
# 图床配置
# 注意在img文件夹所在路径的末端加上一个"/"
location /img/{
    alias {img文件夹所在路径};
    autoindex on;
}
```

3. 设置img的权限

* 首先，nginx必须能读取img/文件夹
* 其次，登陆的用户必须可以用sftp传输图片给img/下

* （假设登录用户名字为userA，userA属于groupA，nginx用户名为www-data，属于组www-data），我们需要设置一个组，使得userA可以读写img/，nginx可以读img/

```bash
# 创建新组
sudo groupadd webadmin

# 将用户加入组
sudo usermod -a -G webadmin userA # 你自己
sudo usermod -a -G webadmin www-data  # Nginx用户

# 更改目录组所有权
sudo chown -R userA:webadmin {img路径}

# 设置权限
# 755表示组用户可以读、写执行权限
sudo chmod -R 2775 /home/ubuntu/img  
```

4. 重启nginx服务

```nginx
sudo nginx -t
sudo systemctl reload nginx
```

### 本地

1. 在自己的本地创建一个json文件，假设命名为PicGoSTFPConfig.json，放在某个位置

```json
{
	"PicGo":{
		"url": "https://{你的域名}",
		"path": "/img/{fullName}",
		"uploadPath": "{PathTofullName, 例如/home/ubuntu/img/{fullName}}",
		"host": "{ip地址}",
		"port": 22,
		"username": "{用户名字}",
		"password": "{用户密码}"
	}
}
```

2. 在PicGo中下载**sftp-uploader**插件

![sftp-uploader](https://img.yumeko.site/file/articles/SFTPImgbedDeployment/sftp-uploader.png)

3. 图床设置

| 图床配置名字 |           随意            |
| :----------: | :-----------------------: |
| **网站标识** | **json文件中你的key名字** |
| **配置文件** |    **json文件的路径**     |
