---
title: SSH Introduction
date: 2025-01-23 
type: post
categories: ["Introduction"]
tags: ["Remote", "Tutorial", "SSH"]
image: ssh-icon.png
# https://www.flaticon.com/
---

# SSH Introduction

SSH is the most commonly used development tool if you have a remote machine, like a supercomputer or a PC in your office. Plentiful introduction to SSH can be found by search engine, like [this one](https://www.geeksforgeeks.org/introduction-to-ssh-secure-shell-key). So I will only record some basic moves commonly used.

## Connect to an SSH server

SSH client is ready-to-use on modern Windows systems.

Suppose you are given a server with:

- hostname: 192.168.31.2
- user: harry
- password: pass-word

The username and password is same as the login information you use to log in that machine on the site. The hostname could be a domain name or raw IP like above.

You are supposed to be able to connect in your shell (of the client machine) with:

```bash
ssh harry@192.168.31.2
```

Ssh client program `ssh` will prompt you to enter the password, then your shell is connected to the remote machine `192.168.31.2`.

Entering the same command-line too often is inefficient, therefore we would like to give `harry@192.168.31.2` a name "office", suppose this connects to the computer in my office.

To record the ssh remote server login, edit the ssh configure file on your client machine `~/.ssh/config` (Windows Also).

``` bash
Host office
    HostName 192.168.31.2
    # Port 22 # Default is 22
    User harry

Host xxxxx
    HostName xxx
    Port xxxx
    User xxxxx
```

After this, typing

```bash
ssh office
```

does the job.

## SSH authentication with keys

You are not allowed to automatically enter the password. So, when connecting SSH with password, you must manually enter it in the console. Moreover, sometimes it might be unsafe to use password.

So, you are supposed to use a pair of public-private keys as ssh's authentication method whenever possible.

If you don't have ssh key pairs, just run in your client machine's shell:

```bash
ssh-keygen -t ed25519
```

and press enter (using all default parameters) when prompted until the end.

The parameter `-t ed25519` chooses `ed25519` as the signature algorithm. It is not recommended to use `rsa`, or you **should actually avoid using `rsa` anymore, [with the reason here](https://blog.trailofbits.com/2019/07/08/fuck-rsa/).**

After `ssh-keygen`, you have the key pair `~/.ssh/id_ed25519` and `~/.ssh/id_ed25519.pub`. **Remember only the `.pub` is the public key and is safe to show.**

Then, you have to inform the ssh server of you public key to be able to connect without a password. Run

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub harry@192.168.31.2
# or equivilantly
ssh-copy-id -i ~/.ssh/id_ed25519.pub office
```

to automatically do the key copying. 

The tool `ssh-copy-id` is not found on Windows, so a manual copying is needed. Just run:

```bash
cat ~/.ssh/id_ed25519.pub
```

on the client machine and copy the output and append it to `~/.ssh/authorized_keys` on the remote (host) machine with:

```bash
echo "your_public_key" >> ~/.ssh/authorized_keys
```

or any text editor you like.

After this, you should be able to simply `ssh office` to enter the remote shell.

## Other parameters

```bash
Host office1
    HostName 192.168.31.3
    User harry
    Port 2222
    ServerAliveCountMax 10
    ServerAliveInterval 45
    IdentityFile ~/.ssh/id_ecdsa
    ProxyCommand "C:\Program Files\Git\mingw64\bin\connect.exe" -a none -S 1.2.3.4:1080 %h %p  
```

- `Port`: Sets the port on the host machine to connect ssh. Default is 22.
- `ServerAliveCountMax`: This sets the maximum number of keepalive messages that can be sent without receiving a response.
- `ServerAliveInterval`: This sets the interval (in seconds) between each keepalive message.
- `IdentityFile`: Explicitly sets the private key to be used.
- `ProxyCommand`: Used to connect to proxy before connecting ssh, in this case, we use a SOCKS5 proxy at `1.2.3.4:1080`.

These parameters can be passed through `-o` option in command line.

For a complete reference, see [Linux man page](https://www.man7.org/linux/man-pages/man5/ssh_config.5.html).

## SSH server on Windows

If you have a Windows machine that you want to access by SSH, you can configure it to be an SSH server. 

Before setting up a server, you should have a little network knowledge. See [Making server visible to the Internet](https://harryzhou2000.github.io/hugo-harry/p/publicinternet/).

### Install OpenSSH server

OpenSSH server feature and its service can be switched on by GUI or command-line on Windows. You can [read this tutorial](https://woshub.com/connect-to-windows-via-ssh/).

Here we have a simple GUI tutorial for Windows 11.

First, install OpenSSH. Search for features from the global search and choose the `optional features` in settings. Then select `Add an optional feature` and search for `openssh`:

![Search features](search-features.png) ![Search OpenSSH](search-openssh.png)

go through until the end and Windows will install OpenSSH:

![Installing](installing-openssh.png)

### Start OpenSSH service

After OpenSSH is installed, go to services (run `services.msc` or search in the bar) and set `OpenSSH SSH Server` and `OpenSSH Authentication Agent`'s `Startup type` to `Automatic` and click OK. If they are not running, start them manually. They will start in the background automatically after next boot.

![Setting ssh service](set-service-ssh.png)

Using `netstat` in CMD will show if the service is listening to the 22 port:

```cmd
C:\Users\harry> netstat -na| find ":22"
TCP    0.0.0.0:22             0.0.0.0:0              LISTENING
...
```

Using this in Windows PowerShell line will see the status of firewall rule:

```powershell
PS C:\Users\harry> Get-NetFirewallRule -Name *OpenSSH-Server* |select Name, DisplayName, Description, Enabled

Name                  DisplayName               Description                                Enabled
----                  -----------               -----------                                -------
OpenSSH-Server-In-TCP OpenSSH SSH Server (sshd) Inbound rule for OpenSSH SSH Server (sshd)    True
```

You can also test if ssh service is running by ssh to `localhost`:

```powershell
ssh harry@localhost
```

Substitute harry with your actual Windows **local account** username. The local account's username is not the same as your Microsoft username, as shown in your `Start` page or `Settings` if you have logged in with a Microsoft Account. You can get your username by checking the path of you home directory. Usually, your home is in `C:\users\<username>`, and with localization like 简体中文 it could look like `C:\用户\<username>` in File Explorer. An easy way is to start a fresh Windows PowerShell or CMD to check the username.

You will be prompted to enter password, which is the **password** and not the **PIN** you use to log in your system.

If you can successfully log in through `localhost`, the ssh service is running.

The configuration of the service is in `C:\Programdata\ssh\sshd_config` by default, and within check the lines:

```bash
PubkeyAuthentication yes
PasswordAuthentication yes
```

to make sure key and password are allowed in the login process.

### Remote SSH access

If you plan to remotely use your computer through SSH, inside or outside the local network, further checking is needed. You need to check the connectivity and use port forwarding + static public IP + static DHCP when necessary, see the [server notes](https://harryzhou2000.github.io/hugo-harry/p/publicinternet/#apply-for-static-ip).

### Firewall rules

You may find by default, `ssh harry@localhost` on the server runs fine, but doing this from any other machine inside or outside the local network gives a connection failure, which is most likely a result of firewall rules.

In the `Windows Defender Firewall with Advanced Security` panel's `inbound` page, find the `OpenSSH` firewall rule we found earlier. Go to `Advanced` tab and check all the `Domain`, `Private` and `Public` profiles, to allow ssh inbound connections to the service from public internet. Click OK to save the changes.

![alt text](ssh-firewall-rules-update.png)

After updating the firewall rule's profile type, try again and this mostly works.

If an existing firewall rule is nonexistent or the method above did not work, try creating an inbound firewall rule in the `Windows Defender Firewall with Advanced Security` panel (I'm using the vanilla Windows Defender, other security solutions like 火绒 or Kaspersky might need different procedures to set the firewall). In the wizard, select `Rule Type` `Port`, select `Protocol and Ports` `TCP` and `Specific local ports` being 22. `Action` is `Allow the connection`. In `Profile` select all. Give a descriptive name like `OpenSSH server inbound rule`.

![Add inbound rule](ssh-firewall.png) ![Wizard](ssh-firewall-wizard.png)

### Key authentication of administrator

If your account is an administrator (which is the most case on Windows), you cannot log in through keys by simply adding your pubkey to `~/.ssh/authorized_keys`. You also need to add them to `C:\ProgramData\ssh\administrators_authorized_keys` (create new file if not existent).
