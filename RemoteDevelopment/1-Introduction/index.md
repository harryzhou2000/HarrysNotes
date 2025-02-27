---
title: Introduction to Remote Developing
date: 2025-01-23 
type: post
categories: ["Introduction"]
tags: ["Remote", "Tutorial"]
image: icon.png
---

According to [ChatGPT](https://chatgpt.com/share/679209a0-eb98-8001-8b0d-b5d9ae2c76fe):

> Remote development has become a cornerstone of modern software engineering, driven by advancements in cloud computing, collaborative tools, and distributed version control systems. It allows developers to work from virtually anywhere while accessing shared resources, codebases, and even powerful remote hardware. Key tools like GitHub, GitLab, and Azure DevOps facilitate seamless collaboration on code, while cloud IDEs such as GitHub Codespaces, Replit, and JetBrains Space enable on-demand development environments that eliminate local setup challenges.
>
> In a remote development workflow, engineers can leverage containerized environments (e.g., Docker) and Infrastructure-as-Code (IaC) to ensure consistency across development, testing, and production. Tools like Slack, Microsoft Teams, and Zoom complement this ecosystem by maintaining communication and team alignment, ensuring agility in workflows.
>
> The shift towards remote development is not only a response to global remote work trends but also an acknowledgment of its benefits, including flexibility, access to diverse talent, and reduced dependency on physical infrastructure. As modern software projects grow increasingly complex, remote development continues to adapt, providing solutions that bridge geographical and technical boundaries while prioritizing productivity and collaboration.

Well, I'm not planning to discuss being a professional software developer, but some key-points about remote developing can be seen: environment, hardware, collaboration ...

Personally, I have been writing CFD related programs, primarily CFD solvers, for academic reasons, and these programs are intended to be executed on PCs, lab servers and supercomputers.

Some running cases of the solver is absurdly large for PCs, so running the program on servers or supercomputers is unavoidable. Sometimes you are forced to test-run these cases to determine bugs. Sometimes it is simply more convenient to write and test the code in the same environment as the running environment.

Therefore, I will record my remote development practice for reference. Hope this will help!