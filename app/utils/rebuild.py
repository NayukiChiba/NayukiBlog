"""
静态页面重建工具
用于在内容更新后自动重建 Astro 静态页面
"""
import os
import subprocess
import threading
import time
from typing import Optional

# 重建状态
_rebuild_lock = threading.Lock()
_is_rebuilding = False
_last_trigger_time: float = 0
_pending_rebuild = False

# 缓存的 frontend 路径
_cached_frontend_dir: Optional[str] = None

# 防抖动时间（秒）- 短时间内多次触发只执行一次
DEBOUNCE_SECONDS = 2.0


def _get_frontend_dir() -> str:
    """获取并缓存 frontend 目录路径"""
    global _cached_frontend_dir
    if _cached_frontend_dir is None:
        fd = os.path.join(os.getcwd(), "frontend")
        _cached_frontend_dir = fd if os.path.exists(fd) else "frontend"
    return _cached_frontend_dir


def trigger_rebuild_async(timeout: int = 300):
    """
    在后台线程中触发前端静态页面重建（带防抖动）

    Args:
        timeout: 构建超时时间（秒），默认 300 秒
    """
    global _last_trigger_time, _pending_rebuild

    current_time = time.time()

    # 防抖动：如果距离上次触发时间太短，标记为待处理
    with _rebuild_lock:
        if current_time - _last_trigger_time < DEBOUNCE_SECONDS:
            _pending_rebuild = True
            print("[Rebuild] 📋 防抖动：已标记待重建")
            return
        _last_trigger_time = current_time
        _pending_rebuild = False

    # 立即启动后台线程
    thread = threading.Thread(target=_run_build, args=(timeout,), daemon=True)
    thread.start()
    print("[Rebuild] 📋 重建任务已加入队列")


def _run_build(timeout: int = 300):
    """实际执行构建的函数"""
    global _is_rebuilding, _pending_rebuild, _last_trigger_time

    with _rebuild_lock:
        if _is_rebuilding:
            print("[Rebuild] ⏳ 已有构建任务在运行，跳过")
            return
        _is_rebuilding = True

    try:
        fd = _get_frontend_dir()
        print(f"[Rebuild] 🚀 开始构建静态页面... ({fd})")

        # 使用 Popen 启动构建进程
        # Linux 上 shell=True 时用字符串命令，Windows 上用列表
        import platform
        is_windows = platform.system() == "Windows"
        cmd = ["npm", "run", "build"] if is_windows else "npm run build"

        process = subprocess.Popen(
            cmd,
            cwd=fd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                print("[Rebuild] ✅ 静态页面构建成功")
            else:
                # 优先显示 stderr，如果为空则显示 stdout
                error_msg = stderr.strip() if stderr and stderr.strip() else stdout.strip() if stdout else 'Unknown error'
                print(f"[Rebuild] ❌ 构建失败 (code={process.returncode}):")
                print(f"[Rebuild] 错误信息: {error_msg[:1000]}")
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            print("[Rebuild] ⚠️ 构建超时")

    except Exception as e:
        print(f"[Rebuild] ❌ 构建错误: {e}")
    finally:
        with _rebuild_lock:
            _is_rebuilding = False

        # 检查是否有待处理的重建请求
        if _pending_rebuild:
            with _rebuild_lock:
                _pending_rebuild = False
                _last_trigger_time = time.time()
            print("[Rebuild] 🔄 处理待定的重建请求...")
            thread = threading.Thread(target=_run_build, args=(timeout,), daemon=True)
            thread.start()


def is_rebuilding() -> bool:
    """检查当前是否正在重建"""
    with _rebuild_lock:
        return _is_rebuilding


def get_rebuild_status() -> dict:
    """
    检查最近一次构建的状态。
    通过检查 dist 目录的修改时间来判断。

    Returns:
        包含构建状态信息的字典
    """
    fd = _get_frontend_dir()
    dist_dir = os.path.join(fd, "dist")

    if os.path.exists(dist_dir):
        mtime = os.path.getmtime(dist_dir)
        last_build = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        return {
            "status": "success",
            "is_rebuilding": is_rebuilding(),
            "last_build": last_build,
            "message": f"最近构建时间: {last_build}"
        }
    else:
        return {
            "status": "not_built",
            "is_rebuilding": is_rebuilding(),
            "message": "尚未构建静态页面，请先运行 npm run build"
        }


def run_rebuild_sync(timeout: int = 300) -> dict:
    """
    同步执行重建（阻塞式）

    Args:
        timeout: 构建超时时间（秒）

    Returns:
        包含构建结果的字典
    """
    try:
        fd = _get_frontend_dir()
        print(f"[Rebuild] 🚀 开始同步构建静态页面... ({fd})")

        # Linux 上 shell=True 时用字符串命令，Windows 上用列表
        import platform
        is_windows = platform.system() == "Windows"
        cmd = ["npm", "run", "build"] if is_windows else "npm run build"

        result = subprocess.run(
            cmd,
            cwd=fd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode == 0:
            print("[Rebuild] ✅ 静态页面构建成功")
            return {
                "status": "success",
                "message": "构建成功"
            }
        else:
            print(f"[Rebuild] ❌ 构建失败: {result.stderr[:300]}")
            return {
                "status": "failed",
                "message": f"构建失败: {result.stderr[:300]}"
            }

    except subprocess.TimeoutExpired:
        print("[Rebuild] ⚠️ 构建超时")
        return {
            "status": "timeout",
            "message": "构建超时"
        }
    except Exception as e:
        print(f"[Rebuild] ❌ 构建错误: {e}")
        return {
            "status": "error",
            "message": f"构建错误: {str(e)}"
        }
