# GitHub 同步教程

本项目的 GitHub 仓库为：

```text
git@github.com:0917Ray/VLM_TTA_Exploration.git
```

以下命令均在项目根目录执行。

## 首次配置

确认 Git 用户信息；如果尚未配置，将示例内容替换为自己的姓名和邮箱：

```bash
git config user.name
git config user.email
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

初始化仓库并关联 GitHub：

```bash
git init -b main
git remote add origin git@github.com:0917Ray/VLM_TTA_Exploration.git
```

可以用下面的命令检查远程地址：

```bash
git remote -v
```

## 日常同步流程

### 1. 查看改动

```bash
git status --short
git diff
```

提交前应确认没有数据集、模型权重、运行结果、密钥或其他大文件。项目的
`.gitignore` 已默认忽略 `data/downloads/`、`data/raw/`、`models/` 和 `runs/`。

查看即将加入版本控制的文件：

```bash
git add --dry-run .
```

### 2. 暂存并提交

```bash
git add .
git status --short
git commit -m "描述本阶段完成的工作"
```

推荐让每次提交只对应一个明确阶段，例如：

```text
Prepare datasets and add visualization review
Add zero-shot CLIP evaluation pipeline
Record baseline evaluation results
```

### 3. 推送到 GitHub

首次推送并建立上游分支：

```bash
git push -u origin main
```

后续推送可以简化为：

```bash
git push
```

## 在同步前获取远程更新

多人或多台机器修改同一仓库时，先获取并以 rebase 方式整合远程提交：

```bash
git fetch origin
git pull --rebase origin main
```

如果出现冲突，打开冲突文件并保留正确内容，然后继续：

```bash
git add <已解决的文件>
git rebase --continue
```

需要取消本次 rebase 时运行：

```bash
git rebase --abort
```

确认本地状态正常后再推送：

```bash
git status
git push
```

## 检查历史

查看最近提交和分支关系：

```bash
git log --oneline --graph --decorate -10
git branch -vv
```

查看某次提交包含的文件和改动统计：

```bash
git show --stat <commit-id>
```

## 大文件与数据集

GitHub 普通 Git 仓库不适合保存原始数据集、压缩包、模型权重和大量实验输出。
本项目只同步复现实验所需的代码、配置与文档；数据和权重应根据
`guide/preparation.md` 在每台机器上单独准备。

如果某个本应忽略的文件已经被 Git 跟踪，仅修改 `.gitignore` 不会自动移除它。
可以只从 Git 索引中取消跟踪，同时保留本地文件：

```bash
git rm --cached <文件路径>
git commit -m "Stop tracking generated file"
```

不要使用 `git push --force` 覆盖共享分支历史。如确实需要改写远程历史，应先确认
影响范围，并优先使用带保护检查的 `git push --force-with-lease`。

## 常见问题

### SSH 身份验证失败

先测试当前 SSH 配置：

```bash
ssh -T git@github.com
```

如果本机还没有 SSH 密钥，按照 GitHub 的 SSH key 文档创建公钥，并将公钥添加到
GitHub 账号。也可以将远程地址切换为 HTTPS：

```bash
git remote set-url origin https://github.com/0917Ray/VLM_TTA_Exploration.git
```

### 推送被拒绝

远程分支包含本地没有的提交时，先整合远程更新再推送：

```bash
git pull --rebase origin main
git push
```
