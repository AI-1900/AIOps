下面整理了一份 Podman 常用命令的表格，包含命令用法、详细说明以及对应的测试用例，供大家参考和实践。

| 命令            | 用法示例                                                      | 详细介绍                                                     | 测试用例说明                                                     |
|-----------------|---------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------|
| **podman pull** | `podman pull <镜像名称>`                                       | 从远程镜像仓库（如 Docker Hub）拉取指定镜像，可指定标签。       | 运行 `podman pull nginx` 后，再使用 `podman images` 查看是否存在 nginx 镜像。 |
| **podman images** | `podman images`                                             | 列出本地存储的所有镜像，显示镜像名称、标签、ID、创建时间及大小等信息。 | 执行 `podman images` 检查当前所有本地镜像。                             |
| **podman run**  | `podman run [OPTIONS] IMAGE [COMMAND]`                         | 创建并启动一个容器。支持后台模式（`-d`）、交互模式（`-it`）、端口映射等。 | 运行 `podman run -d --name mynginx -p 8080:80 nginx`，通过 `podman ps` 验证容器是否在运行。 |
| **podman ps**   | `podman ps [-a]`                                              | 列出正在运行的容器；加 `-a` 参数可以显示所有容器（包括已停止的）。   | 使用 `podman ps` 查看当前正在运行的容器，或 `podman ps -a` 查看全部容器列表。      |
| **podman stop** | `podman stop <容器ID或名称>`                                   | 发送 SIGTERM 信号停止正在运行的容器，适用于正常停止容器服务。         | 运行 `podman stop mynginx` 停止名为 mynginx 的容器。                     |
| **podman rm**   | `podman rm <容器ID或名称>`                                     | 删除已停止的容器，释放容器占用的存储资源。                         | 停止后运行 `podman rm mynginx` 删除容器。                               |
| **podman logs** | `podman logs <容器ID或名称>`                                   | 查看容器的标准输出与错误日志，便于调试和监控容器内部运行状态。         | 使用 `podman logs mynginx` 检查容器运行期间的日志信息。                  |
| **podman exec** | `podman exec -it <容器ID或名称> <命令>`                         | 在正在运行的容器内执行指定命令，可用于进入容器内的交互式 shell。        | 运行 `podman exec -it mynginx bash` 进入容器内，测试交互效果。            |
| **podman build** | `podman build -t <镜像名称> <目录>`                           | 根据 Dockerfile/Containerfile 构建新的镜像，并可为新镜像指定标签。    | 在包含 Dockerfile 的目录下运行 `podman build -t myimage .`，构建完成后用 `podman images` 检查。 |
| **podman commit** | `podman commit <容器ID或名称> <新镜像名称>`                  | 将当前运行或停止的容器状态保存为新的镜像，便于备份或制作定制镜像。      | 运行 `podman commit mynginx mynginx_image`，再用 `podman images` 查看新镜像。 |
| **podman search** | `podman search <关键字>`                                     | 在远程镜像仓库中搜索镜像，返回镜像名称、描述等信息。                  | 执行 `podman search nginx`，查看返回的 nginx 相关镜像列表。             |
| **podman push** | `podman push <镜像名称> [目标仓库地址]`                          | 将本地镜像推送到远程镜像仓库，实现镜像共享和跨环境部署。               | 例如：`podman push myimage docker.io/username/myimage`，推送成功后可在仓库中查看。 |

---

### 说明

- **podman pull** 与 **podman search**：帮助用户获取需要的镜像。
- **podman run**、**ps**、**stop**、**rm**：覆盖了容器的生命周期管理。
- **podman logs** 与 **exec**：方便在容器运行过程中进行调试。
- **podman build** 与 **commit**：用于构建或保存镜像，满足定制化需求。
- **podman push**：适用于将本地镜像发布到远程仓库，方便镜像共享。

以上命令均可在 Linux 终端下直接执行，建议根据实际应用场景结合相应参数进行测试。通过不断实践，您将更加熟悉 Podman 的管理与操作流程。