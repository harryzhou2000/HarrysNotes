---
title: Using EasyConnect To Proxy Connection to a Supercomputer
date: 2025-02-18T23:35:20+08:00
type: post
categories: ["Introduction"]
tags: ["Remote", "Tutorial", "SSH", "EasyConnect"]
image: image.png
# https://www.flaticon.com/

---
<!-- ![alt text](image.png) -->
# Using EasyConnect To Proxy Connection to a Supercomputer

## Sangfor EasyConnect

I need to use `EasyConnect` as a proxy to connect to a supercomputer.

The supercomputer provider gives IP and login information of SSH through the proxy. 

The `EasyConnect` software runs a proxy client and listens to a port (`1080` by default) as SOCKS5 proxy endpoint.

Sadly, on Windows, `EasyConnect` introduces unnecessary root certificate in your Windows certificate system, and runs suspicious background services even the proxy is shutdown. 

See [how to uninstall `EasyConnect`](https://blog.csdn.net/m0_52116878/article/details/139015155)

## Docker

We can use a docker environment to safely run EasyConnect.

See the [Github repo for a EasyConnect Docker Image](https://github.com/docker-easyconnect/docker-easyconnect)

The image is `hagb/docker-easyconnect`, you may need some docker image to access it.

Simply use this bash script:

```bash
docker run --name easyConnect -d --restart always \
        --device /dev/net/tun --cap-add NET_ADMIN -ti \
        -e PASSWORD=xxxx -e URLWIN=1 \
        -v $HOME/.ecdata:/root \
        -p 127.0.0.1:5901:5901 -p 0.0.0.0:1080:1080 -p 127.0.0.1:8888:8888 \
        docker.1ms.run/hagb/docker-easyconnect
```

then the container will run in background.

A VNC viewer can access `127.0.0.1:5901` on the host. Password for VNC is `xxxx` as specified in the arguments. 

If your host is Linux, you can use `xtigervncviewer`. On Windows there is `RealVNC`.

If your Linux host is connected via SSH, use X11 forwarding:

```bash
ssh -Y user@hostname
```

On Windows, [you can use `MobaXterm ` or `PuTTY & xming`.](https://it.engineering.oregonstate.edu/run-x11-application-windows)

But I would recommend using WSLg's graphic capabilities if you have WSL2 on your Windows.

Inside WSL2, set up your ssh host correctly and connect via `ssh -Y ...`.

In the X11 forwarding enabled SSH connection, running `xtigervncviewer` will open a graphic window tunneled from the host, allowing you to access `EasyConnect` with GUI.

Without VNC, [you can also use a web UI.](https://github.com/docker-easyconnect/docker-easyconnect/blob/master/doc/usage.md#web-%E7%99%BB%E5%BD%95)

## SSH through SOCKS5

To run your ssh through SOCKS5, on Linux, use configuration:

``` xml
Host <name>
    HostName <hostname>
    User  <user>
    Port 22
    ProxyCommand  nc -X 5 -x 127.0.0.1:1080 %h %p
```

On Windows:

```xml
Host <name>
    HostName <hostname>
    User <user>
    ProxyCommand "C:\Program Files\Git\mingw64\bin\connect.exe" -a none -S localhost:31080 %h %p    
```

Where "C:\Program Files\Git\mingw64\bin\connect.exe" is a tool ported with `Git for Windows`.

## Keeping alive

Sometimes `EasyConnect` docker container dies (maybe due to being idle). 

Luckily, simply restarting the container would fix it because the login credentials are persistent.

We can use a script to check connectivity and restart docker container if needed:

`restoreConnectivity.sh`:

```bash
#!/bin/bash

touch /tmp/scp
scp -o "ProxyCommand=nc -X 5 -x 127.0.0.1:1080 %h %p" \
        -o "IdentityFile=/path/to/id_xxx" \
        -o StrictHostKeyChecking=accept-new -v \
        /tmp/scp user@host:~/scp_test

if [[ $? -eq 0 ]]
then
        echo "$(date)        easyConnect is good "
else
        # echo "=================================="
        echo "$(date)        easyConnect not working; restarting    "
        # echo "=================================="
        docker restart easyConnect
fi
```

To periodically run this script, you can use cron with is commonly ported with `systemd`. 

Use

```bash
sudo systemd status cron
```

to check is cron service is running.

Edit your user crontab with

```bash
crontab -e
```

And add a line like:

``` bash
*/3 * * * * bash ~/path/to/restoreConnectivity.sh 1>> ~/.log/cron.log 2>> ~/.log/cron_stderr.log
```

which means every 3 minutes the script `restoreConnectivity.sh` is executed and output is appended to `~/.log/cron.log` and `~/.log/cron_stderr.log`. 

To avoid the log being infinitely growing, add a log cleaning script:

`refreshCronLog.sh`:

```bash
#!/bin/bash

touch ~/.log/cron.log; touch ~/.log/cron_stderr.log
mv ~/.log/cron.log ~/.log/cron.log.bkp
mv ~/.log/cron_stderr.log ~/.log/cron_stderr.log.bkp
tail -n 1024 ~/.log/cron.log.bkp > ~/.log/cron.log
tail -n 1024 ~/.log/cron_stderr.log.bkp > ~/.log/cron_stderr.log
```

then add another line with `crontab -e`:

```bash
0 0 * * * bash ~/path/to/refreshCronLog.sh
```

Which means every day at 00:00:00 the `refreshCronLog.sh` is executed to keep the log files short.

## PS

Exposing the SOCKS5 proxy endpoint to public network is **dangerous**. Do not do that.
