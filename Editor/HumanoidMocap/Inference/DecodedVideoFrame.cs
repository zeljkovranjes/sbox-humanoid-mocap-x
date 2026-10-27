using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace HumanoidMocapX.EditorTools.Inference;

public sealed record DecodedVideoFrame(byte[] Rgba,int Width,int Height,double Time);
