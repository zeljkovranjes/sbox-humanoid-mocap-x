using System.Runtime.InteropServices;
using HumanoidMocapX.EditorTools.Inference;

static class VideoLayoutTests
{
    internal static void Run()
    {
        Reject(() => new VideoFrameLayout(8193,1,0,0,8193,1,0));
        Reject(() => new VideoFrameLayout(1,8193,0,0,1,8193,0));
        Reject(() => new VideoFrameLayout(8192,8192,0,0,8192,8192,0));
        Reject(() => new VideoFrameLayout(2,2,1,0,2,2,0));
        Reject(() => new VideoFrameLayout(2,2,0,0,2,2,45));
        _=new VideoFrameLayout(7680,4320,0,0,7680,4320,0);
        _=new VideoFrameLayout(8192,1,0,0,8192,1,0);
        var data=Marshal.AllocHGlobal(16);
        try
        {
            Marshal.Copy(new byte[]{1,2,3,0,4,5,6,0,7,8,9,0,10,11,12,0},0,data,16);
            var layout=new VideoFrameLayout(2,2,0,0,2,2,0);
            Check(layout.CopyRgba(data,16,8,default),3,6,9,12);
            Check(layout.CopyRgba(data,16,-8,default),9,12,3,6);
            Check(new VideoFrameLayout(2,2,0,0,2,2,90).CopyRgba(data,16,8,default),9,3,12,6);
            Reject(()=>layout.CopyRgba(data,16,4,default));
            Reject(()=>layout.CopyRgba(data,8,8,default));
            Reject(()=>layout.CopyRgba(IntPtr.Zero,16,8,default));
            Reject(()=>layout.CopyRgba(data,int.MaxValue,int.MaxValue,default));
            Reject(()=>layout.CopyRgba(data,int.MaxValue,int.MinValue,default));
            using var cancellation=new CancellationTokenSource(); cancellation.Cancel();
            try { layout.CopyRgba(data,16,8,cancellation.Token); throw new Exception("Cancelled video copy was accepted"); }
            catch(OperationCanceledException) { }
        }
        finally { Marshal.FreeHGlobal(data); }
        Console.WriteLine("PASS: video dimension/stride bounds, BGR conversion, bottom-up and rotated frames");
    }

    static void Check(byte[] rgba,params byte[] red)
    {
        MotionTests.Check(rgba.Length==red.Length*4,"Unexpected RGBA dimensions");
        for(var i=0;i<red.Length;i++)
            MotionTests.Check(rgba[i*4]==red[i]&&rgba[i*4+3]==255,"Video orientation or RGBA channel order changed");
    }

    static void Reject(Action action)
    {
        try { action(); } catch(InvalidDataException) { return; }
        throw new Exception("Invalid video header or buffer was accepted");
    }
}
