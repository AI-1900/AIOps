## Git常见用法、测试用例及详细介绍

### 1. Git 基本操作
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 初始化仓库 | `git init` | 在当前目录初始化一个新的Git仓库 | 在一个新文件夹中执行 `git init`，检查是否生成 `.git` 目录 |
| 克隆仓库 | `git clone <repo_url>` | 从远程仓库克隆到本地 | `git clone https://github.com/example/repo.git` |
| 配置用户信息 | `git config --global user.name "Your Name"` `git config --global user.email "your@email.com"` | 设置全局用户名和邮箱 | 执行 `git config --list` 查看配置 |

---

### 2. Git 分支操作
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 查看分支 | `git branch` | 列出本地所有分支 | 在已有仓库运行 `git branch` |
| 创建分支 | `git branch <branch_name>` | 创建新分支 | `git branch feature`，然后 `git branch` 查看 |
| 切换分支 | `git checkout <branch_name>` 或 `git switch <branch_name>` | 切换到指定分支 | `git checkout feature` |
| 创建并切换分支 | `git checkout -b <branch_name>` 或 `git switch -c <branch_name>` | 创建新分支并切换 | `git checkout -b new_feature` |
| 合并分支 | `git merge <branch_name>` | 合并指定分支到当前分支 | 在 `main` 分支执行 `git merge feature` |

---

### 3. Git 提交与日志管理
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 查看状态 | `git status` | 查看工作区状态 | 修改文件后执行 `git status` |
| 添加文件 | `git add <file>` 或 `git add .` | 添加指定文件或所有变更到暂存区 | `git add test.txt` |
| 提交更改 | `git commit -m "message"` | 提交暂存区文件到本地仓库 | `git commit -m "Initial commit"` |
| 查看提交历史 | `git log` | 显示提交历史 | `git log --oneline` |

---

### 4. Git 远程仓库操作
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 查看远程仓库 | `git remote -v` | 查看远程仓库列表 | `git remote -v` |
| 添加远程仓库 | `git remote add origin <url>` | 绑定远程仓库 | `git remote add origin https://github.com/user/repo.git` |
| 推送到远程 | `git push origin <branch>` | 推送本地分支到远程 | `git push origin main` |
| 拉取最新代码 | `git pull origin <branch>` | 拉取远程分支最新代码 | `git pull origin main` |

---

### 5. Git 标签管理
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 创建标签 | `git tag <tag_name>` | 创建轻量标签 | `git tag v1.0` |
| 删除标签 | `git tag -d <tag_name>` | 删除本地标签 | `git tag -d v1.0` |
| 推送标签 | `git push origin <tag_name>` | 推送标签到远程仓库 | `git push origin v1.0` |

---

### 6. Git 撤销与回滚
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 撤销文件修改 | `git checkout -- <file>` | 丢弃工作区修改 | 修改 `test.txt` 后 `git checkout -- test.txt` |
| 取消暂存 | `git reset HEAD <file>` | 取消已 `git add` 的文件 | `git add test.txt` 后 `git reset HEAD test.txt` |
| 回滚到指定提交 | `git reset --hard <commit_id>` | 强制回滚到某个提交 | `git reset --hard HEAD~1` |

---

### 7. Git 其他常见命令
| 操作 | 命令 | 详细介绍 | 测试用例 |
|------|------|----------|----------|
| 显示差异 | `git diff` | 比较未暂存的修改 | 修改 `test.txt` 后执行 `git diff` |
| 显示提交的文件修改 | `git show <commit_id>` | 显示某次提交的修改内容 | `git show HEAD` |
| 清理未追踪文件 | `git clean -f` | 删除未追踪文件 | `git clean -f` |

---

以上是 Git 的常见操作及其测试用例，可以根据具体需求进行拓展和深入学习。