---
title: Making server visible to the Internet
date: 2025-01-23 
type: page
categories: ["Introduction"]
tags: ["Tutorial", "Server"]
# image: ssh-icon.png
slug: PublicInternet
# https://www.flaticon.com/
---

# Making server visible to the Internet

Computers connected to the Internet are generally speaking connected to each other. However, if you intend to host your own site or provide any internet service and access it anywhere without relying on some other people's resource, you will need to expose your server machine to the public internet.

To put it simply, you can only connect to a server that is *"visible"* to the client. If your client and server are in the same local network, and your server is assigned a static ip `192.168.31.2` in the DHCP, you can see the server though the net. If your client machine is not in the local net, the IP address `192.168.31.2` does not direct to the server, because you and your server should be connected via the public internet. The server can access the public internet because of the NAT service on the router.

## Relay with cloud service

You can set up a cloud machine with static public IP and establish connection, for example SSH tunnel, from your server, so that the server can be accessed with the static public IP. You can also set up reverse proxy application through the cloud machine to expose the service to public internet. See [frp](https://github.com/fatedier/frp) for example.

## Apply for static IP

You can ask your ISP, or your organization if you are in a research institution or company to assign a static public IP to your internet access. Usually the device connected to a cable from ISP directly gets the static public IP. If your cable is connected to the WAN port of a router (usually with wifi capabilities), the router has static public IP. In the latter case, your server's IP is within the local network allocated by the router, then you can use port forwarding to expose ports to the public internet.

## Other methods

You can use some online 3rd party solutions to expose your service, like:

- [ngrok](https://ngrok.com/)
- [贝瑞花生壳](https://hsk.oray.com/)
- [ZeroTier](https://www.zerotier.com/)

They are generally referred
These can be more stable but has, more or less, a price tag.

Interesting tools or tutorials:

- [pwnat](https://github.com/samyk/pwnat)
- [frp tutorial](https://hyabc.github.io/frp-tutorial/)
