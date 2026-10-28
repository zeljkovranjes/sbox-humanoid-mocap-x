# Processing machine over SSH

**Processing machine (SSH)** runs captures on another Linux computer with the
standalone worker (`services/humanoid-mocap-worker`). The editor's computer
only copies the video there and imports the motion, so a capture does not load
its GPU. Starting `ssh` is an exception to the library's no-external-process
rule, made because users asked to process on their own GPU machine.

## Requirements

- An OpenSSH client on this computer, with `ssh` on `PATH`. Windows 10 and
  later include one.
- Key-based sign-in. `ssh -o BatchMode=yes user@host true` must succeed
  without a prompt: the editor never answers password or host-key prompts, so
  connect once in a terminal to trust the host key.
- On the machine: the worker, its models (`humanoid-mocap-worker models`) and
  FFmpeg (`ffmpeg` and `ffprobe` on `PATH`). The default worker path is
  `~/humanoid-mocap-worker/humanoid-mocap-worker`.

## Setup

In **Mocap adjustments › Processing › Setup**, choose **Processing machine
(SSH)**, enter the address (`user@host`, or a host alias from
`~/.ssh/config`) and the worker's path, then press **Check**. Check connects,
confirms that FFmpeg is installed and that the worker starts, and then saves
the choice. Local model downloads are not needed. The address and path are
remembered in the `HumanoidMocapX.SshAddress` and `HumanoidMocapX.SshWorker`
editor cookies.

**Motion model** and **Camera sampling** apply to the worker. The worker
chooses its own parallel frames for the machine's GPU.

## A capture

Each capture opens four ssh sessions:

1. Create `/tmp/humanoid-mocap-x-<random>` (mode 700) and stream the video into
   it as `video.<ext>`. The user's file name, which may contain quotes or
   spaces, never reaches the machine; every remote argument is single-quoted.
2. Run `humanoid-mocap-worker capture` there and forward its JSON progress
   lines to the status area.
3. Stream `capture.hmotion` back into `HumanoidMocapX/captures`, validate it
   and point it at the local video.
4. Remove the job directory.

A failure shows ssh's own message or the last lines of the worker's standard
error, without .NET stack frames.

## Cancellation

The worker runs under a small `sh` wrapper whose standard input is the ssh
session. Cancelling closes that input, and so does a dropped connection or the
editor exiting. When the input ends, the wrapper sends the worker `SIGTERM`,
then `SIGKILL` 10 s later if it is still running, and removes the job
directory. The wrapper ignores `SIGPIPE`, so a dropped session cannot stop it
before then. The editor waits up to 15 s for the session to end before it cuts
it.

## Limitations

- One capture at a time per editor window.
- A job directory stays in `/tmp` if the editor exits between the capture and
  the download, until the machine clears `/tmp`.
- Clips are limited to 5,000 frames, as on this computer.
