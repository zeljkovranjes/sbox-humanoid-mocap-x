using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Editor;
using Sandbox;
using HumanoidMocapX.Core.Preview;
using HumanoidMocapX.EditorTools.Video;

namespace HumanoidMocapX.EditorTools;

/// <summary>Timestamped C# video preview. One decoder and a bounded frame handoff;
/// decoding never runs on the editor thread.</summary>
sealed class MocapVideoWidget : Widget
{
    public Action TogglePlayback { get; set; }
    public double Duration { get; private set; }
    public double? FrameTime { get; private set; }
    public string Error { get; private set; }
    readonly CancellationTokenSource stop=new();
    readonly object sync=new();
    double requested,decodedRequest=double.NaN;
    bool playEngineVideo;
    DecodedVideoFrame pending;
    string failure;
    double duration;
    Pixmap frame;
    readonly EngineVideoPreview enginePreview;
    readonly Action<PreviewFrameBuffer,double> receiveEngineFrame;

    public MocapVideoWidget(Widget parent,string path,bool useEngineVideo=false):base(parent)
    {
        MinimumSize=50;
        if(useEngineVideo)
        {
            receiveEngineFrame=ReceiveEngineFrame;enginePreview=new EngineVideoPreview(path);
            stop.Dispose();return;
        }
        _=Task.Run(()=>Decode(path));
    }
    public void Request(double seconds,bool playRealtime=false)
    {lock(sync){requested=Math.Max(0,seconds);playEngineVideo=playRealtime;}}

    void Decode(string path)
    {
        try
        {
            var metadata=Mp4Metadata.Read(path);
            lock(sync)duration=metadata.Duration;
            using var decoder=new WindowsVideoDecoder(path);
            var cache=new Dictionary<long,DecodedVideoFrame>();var order=new Queue<long>();
            static long Key(double time)=>(long)Math.Round(time*10000);
            DecodedVideoFrame Read()
            {
                var decoded=decoder.Read(stop.Token);if(decoded is null)return null;
                var visible=decoded.Height;
                var scale=Math.Min(1,Math.Min(960d/decoded.Width,540d/visible));
                var width=Math.Max(1,(int)(decoded.Width*scale));var height=Math.Max(1,(int)(visible*scale));
                var rgba=new byte[checked(width*height*4)];
                for(var y=0;y<height;y++)
                {
                    stop.Token.ThrowIfCancellationRequested();var sourceY=y*visible/height;
                    for(var x=0;x<width;x++)
                    {
                        var source=(sourceY*decoded.Width+x*decoded.Width/width)*4;var destination=(y*width+x)*4;
                        rgba[destination]=decoded.Rgba[source];rgba[destination+1]=decoded.Rgba[source+1];
                        rgba[destination+2]=decoded.Rgba[source+2];rgba[destination+3]=255;
                    }
                }
                var preview=new DecodedVideoFrame(rgba,width,height,decoded.Time);var key=Key(preview.Time);
                if(!cache.ContainsKey(key))
                {
                    // At most 32 preview frames and 64 MiB; raw inference frames are never changed.
                    var limit=Math.Max(1,Math.Min(32,64*1024*1024/rgba.Length));
                    while(cache.Count>=limit)cache.Remove(order.Dequeue());
                    order.Enqueue(key);cache[key]=preview;
                }
                return preview;
            }
            DecodedVideoFrame current=null,next=null;
            while(!stop.IsCancellationRequested)
            {
                double target;lock(sync)target=requested;
                if(target==decodedRequest){stop.Token.WaitHandle.WaitOne(8);continue;}
                var index=Array.BinarySearch(metadata.Times,target+.00001);
                if(index<0)index=~index-1;index=Math.Clamp(index,0,metadata.Times.Length-1);
                if(cache.TryGetValue(Key(metadata.Times[index]),out var cached))
                {lock(sync){pending=cached;decodedRequest=target;}continue;}
                if(current is not null&&(target<current.Time-.00001||target>current.Time+1))
                {decoder.Seek(target);current=null;next=null;}
                current??=Read();
                if(current is null)throw new InvalidOperationException("The video contains no decodable frames.");
                while(true)
                {
                    next??=Read();
                    if(next is null||next.Time>target+.00001)break;
                    current=next;next=null;
                    // A new scrub request supersedes this one; no queue of seeks builds up.
                    lock(sync)if(Math.Abs(requested-target)>.1)break;
                }
                lock(sync)
                {
                    if(Math.Abs(requested-target)<=.1)pending=current;
                    decodedRequest=target;
                }
            }
        }
        catch(OperationCanceledException) { }
        // An iPhone HEVC video on a PC without Microsoft's HEVC Video Extensions is converted before capture; the converted
        // copy replaces this preview then.
        catch(NotSupportedException){lock(sync)failure="Preparing a preview of this video… It is converted to a format this PC can play when it is processed.";}
        catch(Exception e){lock(sync)failure="Video preview: "+e.Message;}
        finally{stop.Dispose();}
    }
    public void Present()
    {
        if(enginePreview is not null)
        {
            double time;bool playing;lock(sync){time=requested;playing=playEngineVideo;}
            enginePreview.Present(time,playing,receiveEngineFrame);Duration=enginePreview.Duration;
            var error=enginePreview.Error;if(Error!=error){Error=error;Update();}
            return;
        }
        DecodedVideoFrame decoded;var previousError=Error;
        lock(sync){decoded=pending;pending=null;Duration=duration;Error=failure;}
        if(Error!=previousError)Update();
        if(decoded is null||FrameTime==decoded.Time)return;
        // The decoder supplies the same visible, oriented image used by inference.
        var size=new Vector2(decoded.Width,decoded.Height);
        if(frame is null||frame.Size!=size)frame=new Pixmap(size);
        frame.UpdateFromPixels(decoded.Rgba,size,ImageFormat.RGBA8888);
        FrameTime=decoded.Time;Update();
    }
    void ReceiveEngineFrame(PreviewFrameBuffer pixels,double time)
    {
        var size=new Vector2(pixels.Width,pixels.Height);
        if(frame is null||frame.Size!=size)frame=new Pixmap(size);
        if(!frame.UpdateFromPixels(pixels.Pixels,size,ImageFormat.RGBA8888))
            throw new InvalidOperationException("The editor could not display the decoded video frame.");
        FrameTime=time;Update();
    }
    internal byte[] FramePng()=>frame?.GetPng();
    protected override void OnPaint()
    {
        Paint.ClearPen();Paint.SetBrush(Theme.ControlBackground);Paint.DrawRect(LocalRect);
        if(frame is null)
        {
            if(Error is not null){Paint.SetPen(Theme.TextLight);Paint.DrawText(LocalRect.Shrink(12),Error,TextFlag.Center|TextFlag.WordWrap);}
            return;
        }
        var scale=Math.Min(Width/frame.Size.x,Height/frame.Size.y);
        var size=frame.Size*scale;
        var picture=new Rect((Width-size.x)*.5f,(Height-size.y)*.5f,size.x,size.y);
        Paint.Draw(picture,frame);
    }
    protected override void OnMouseClick(MouseEvent e)
    {base.OnMouseClick(e);if(e.LeftMouseButton)TogglePlayback?.Invoke();}
    public override void OnDestroyed()
    {
        if(enginePreview is not null)enginePreview.Dispose();
        else try{stop.Cancel();}catch(ObjectDisposedException){}
        base.OnDestroyed();
    }
}
