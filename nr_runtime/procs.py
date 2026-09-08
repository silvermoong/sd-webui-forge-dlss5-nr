"""Bind owned descendants to their controller with a Windows kill-on-close Job.

Kernel ownership also applies after an abrupt controller exit. Unrelated
processes are never enrolled, stopped or unloaded by this module.
"""
import ctypes
import os
import subprocess
import sys
import threading
from typing import Optional

_WINDOWS = sys.platform == "win32"
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9
_PROCESS_SET_QUOTA, _PROCESS_TERMINATE = 0x0100, 0x0001

_job: Optional[int] = None
_lock = threading.Lock()


if _WINDOWS:
    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", ctypes.c_ulong),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_ulong),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", ctypes.c_ulong),
                    ("SchedulingClass", ctypes.c_ulong)]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]


def _kernel32():
    """句柄是 64 位的，**必须设 restype** —— ctypes 默认按 c_int 取返回值，
    会把句柄截成 32 位，之后 SetInformationJobObject / AssignProcessToJobObject
    拿到的就是个野值，而 CloseHandle(野值) 有可能关掉本进程别的句柄。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_ulong]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    return k32


def job() -> Optional[int]:
    """本进程那个 job 的句柄。**故意不关它** —— 句柄一关 job 就生效把孩子全杀了，
    我们要的正是「进程消失时才发生」。"""
    global _job
    if not _WINDOWS:
        return None
    with _lock:
        if _job is None:
            k32 = _kernel32()
            h = k32.CreateJobObjectW(None, None)
            if not h:
                return None
            info = _EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            k32.SetInformationJobObject(h, _JobObjectExtendedLimitInformation,
                                        ctypes.byref(info), ctypes.sizeof(info))
            _job = h
    return _job


def adopt(pid: int) -> bool:
    """把一个已经在跑的进程收进 job。孙进程跟着一起被收（bat 起的那些就是孙）。"""
    h = job()
    if not h:
        return False
    k32 = _kernel32()
    proc = k32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
    if not proc:
        return False
    try:
        return bool(k32.AssignProcessToJobObject(h, proc))
    finally:
        k32.CloseHandle(proc)


_PATHEXT = ".COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC"


def launch_env() -> dict:
    """给子进程用的环境块：本进程那份，修掉 git bash 弄坏的几处。

    **只补不覆盖**：自己配过的 PATHEXT 照用，只有连 `.EXE` 都不认的那种才判定为坏了。
    """
    # NoDefaultCurrentDirectoryInExePath：git bash 会设它，cmd 见了就不再先找当前目录。
    # 启动 bat 普遍靠这条默认行为（forge 的 `webui-user.bat` 最后一行是 `call webui.bat`），
    # 传下去它当场报「不是内部或外部命令」。删的时候按大小写不敏感找 ——
    # Windows 上 `os.environ` 的键是全大写的
    env = {k: v for k, v in os.environ.items()
           if k.lower() != "nodefaultcurrentdirectoryinexepath"}
    if not _WINDOWS:
        return env
    if ".exe" not in env.get("PATHEXT", "").lower():
        env["PATHEXT"] = _PATHEXT
    root = env.get("SYSTEMROOT") or env.get("WINDIR") or r"C:\Windows"
    env.setdefault("OS", "Windows_NT")
    if not env.get("COMSPEC"):
        env["COMSPEC"] = os.path.join(root, "System32", "cmd.exe")
    return env


def spawn(*args, **kwargs) -> subprocess.Popen:
    """`subprocess.Popen` 的替身，起完顺手收进 job。收不进也照常返回 ——
    进程绑定失败不该让功能整个不可用。

    **默认显式传 `launch_env()`**（本进程那份环境，修掉 git bash 弄坏的几处）。
    本进程的真实环境块被原生库改过，跟 `os.environ` 对不上：实测 `os.environ["PATH"]`
    里有 `C:\\Program Files\\Git\\cmd`，而 `subprocess.run(["git", "--version"])`
    报 WinError 2。继承那份坏环境的后果是 forge 的 bat 里 GitPython 报
    「Bad git executable」、bat 停在 `pause` 上不动，表现成「拉起来了但永远不就绪」。

    ⚠️ 传 env **修不好「找不到 exe」**：`CreateProcess` 解析可执行文件名用的永远是
    父进程的真实环境块。所以启动命令要给绝对路径。
    """
    kwargs.setdefault("env", launch_env())
    p = subprocess.Popen(*args, **kwargs)
    adopt(p.pid)
    return p
