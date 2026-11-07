# GitHub 同步

本指南只同步代码、配置和文档。数据、模型权重和实验输出保留在本机，不能提交。

仓库远程地址：

```text
git@github.com:0917Ray/VLM_TTA_Exploration.git
```

## 首次设置

在仓库根目录查看身份和远程：

```bash
git config user.name
git config user.email
git remote -v
```

如尚未配置身份：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

已有本地 Git 仓库时不要再次执行 `git init`。仅当 `git remote -v` 没有 `origin` 时添加远程：

```bash
git remote add origin git@github.com:0917Ray/VLM_TTA_Exploration.git
```

## 日常提交

先检查状态和差异：

```bash
git status --short
git diff
git add --dry-run .
```

确认没有数据、模型、运行结果、密钥或大文件后再提交：

```bash
git add README.md guide configs scripts pyproject.toml requirements.txt
git status --short
git commit -m "Describe the change"
```

按实际修改调整 `git add` 路径；提交前检查暂存内容。

首次推送：

```bash
git push -u origin main
```

之后：

```bash
git push
```

## 获取远端更新

推送前先整合远端变更：

```bash
git fetch origin
git pull --rebase origin main
```

发生冲突时，编辑冲突文件并确认内容，然后：

```bash
git add README.md
git rebase --continue
```

取消本次 rebase：

```bash
git rebase --abort
```

确认状态后再推送：

```bash
git status
git push
```

## 文件保护

`.gitignore` 已忽略 `data/downloads/`、`data/raw/`、`models/` 和 `runs/`。已被 Git 跟踪的文件不会因新增 ignore 规则自动取消跟踪；需要从索引移除但保留本地文件时：

```bash
git rm --cached README.md
```

不要用 `git push --force` 覆盖共享历史。确需改写时，先确认影响范围并使用 `--force-with-lease`。

SSH 验证失败时可检查：

```bash
ssh -T git@github.com
```

也可以改用 HTTPS：

```bash
git remote set-url origin https://github.com/0917Ray/VLM_TTA_Exploration.git
```
