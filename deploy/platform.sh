#!/usr/bin/env bash
# Existing local image only. No pull, build, package, host install or port killing.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
name="${FS_CONTAINER_NAME:-fs-aircraft-platform}"
image="${FS_IMAGE:-fs-aircraft:prototype}"
port="${FS_PORT:-8002}"
resources="${FS_RESOURCES:-backend/resources}"
state="${FS_STATE:-backend/exp/platform}"
[[ "$resources" == /* ]] || resources="$root/$resources"
[[ "$state" == /* ]] || state="$root/$state"
owner_key=org.federatedscope.platform.root
signature_key=org.federatedscope.platform.settings
fail() { printf '%s\n' "$*" >&2; exit 1; }
command -v docker >/dev/null || fail '未安装 Docker；本脚本不会自动安装宿主机软件。'
docker info >/dev/null 2>&1 || fail 'Docker 引擎不可用或当前用户无权限。'
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ ]] || fail '容器名称非法。'
owned() {
    local recorded
    recorded="$(docker inspect --format "{{ index .Config.Labels \"$owner_key\" }}" "$name")"
    [[ "$recorded" == "$root" ]] || fail '同名容器不属于本目录，拒绝操作。'
}
case "${1:-check}" in
    stop|status|logs)
        owned
        case "$1" in
            stop) docker stop --time 30 "$name" ;;
            status) docker inspect --format '{{json .State}}' "$name" ;;
            logs) docker logs --tail 200 "$name" ;;
        esac
        exit ;;
    check|start) ;;
    *) fail '用法：bash backend/deploy/platform.sh check|start|stop|status|logs' ;;
esac
[[ "$port" =~ ^[0-9]{1,5}$ ]] || fail 'FS_PORT 必须是 1–65535。'
port=$((10#$port))
(( port >= 1 && port <= 65535 )) || fail 'FS_PORT 必须是 1–65535。'
[[ -d "$resources" ]] || fail "资源目录缺失：$resources"
[[ -d "$state" ]] || fail "先创建结果目录并确认权限：$state"
resources="$(cd -- "$resources" && pwd -P)"
state="$(cd -- "$state" && pwd -P)"
[[ "$resources" != *,* && "$state" != *,* ]] || fail 'Docker 挂载路径不能含逗号。'
[[ -w "$state" ]] || fail '结果目录对当前用户不可写。'
image_id="$(docker image inspect --format '{{.Id}}' "$image")" || fail '本地镜像不存在；等待正式打包后先 docker load，本脚本不联网拉取。'
runtime=(--pull never --gpus device=0 --user "$(id -u):$(id -g)" --read-only
    --cap-drop ALL --security-opt no-new-privileges --init --shm-size 1g
    --tmpfs /tmp:rw,nosuid,size=1g
    --mount "type=bind,source=$resources,target=/opt/platform/backend/resources,readonly"
    --mount "type=bind,source=$state,target=/opt/platform/backend/exp/platform")
printf '%s\n' '正在执行断网 GPU 自检、三方法预检、原图与缓存校验（不训练）…'
docker run --rm --network none "${runtime[@]}" --entrypoint python "$image_id" \
    scripts/check_platform_deployment.py --gpu --require-cache
[[ "${1:-check}" == start ]] || exit 0
signature="$(printf '%s\n' "$image_id" "$resources" "$state" "$port" "$(id -u):$(id -g)" | sha256sum | cut -d ' ' -f 1)"
if docker container inspect "$name" >/dev/null 2>&1; then
    owned
    recorded="$(docker inspect --format "{{ index .Config.Labels \"$signature_key\" }}" "$name")"
    [[ "$recorded" == "$signature" ]] || fail '已有容器的镜像/挂载/端口已改变；请使用新的 FS_CONTAINER_NAME，避免覆盖原容器。'
    docker start "$name"
else
    docker run -d "${runtime[@]}" --name "$name" --stop-timeout 30 \
        --label "$owner_key=$root" --label "$signature_key=$signature" \
        -p "127.0.0.1:$port:8002" "$image_id"
fi
for attempt in {1..60}; do
    health="$(docker inspect --format '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$name")"
    case "$health" in
        running/healthy)
            printf '服务地址：http://127.0.0.1:%s；状态：bash backend/deploy/platform.sh status\n' "$port"
            exit 0 ;;
        exited/*|dead/*|running/unhealthy)
            fail '服务未通过健康检查；容器和日志已保留，请运行 logs 查看。' ;;
    esac
    sleep 1
done
fail '服务健康检查超时；容器和日志已保留，请运行 status/logs 查看。'
