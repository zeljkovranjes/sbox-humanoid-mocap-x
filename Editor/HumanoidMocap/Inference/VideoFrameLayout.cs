using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace HumanoidMocapX.EditorTools.Inference;

/// <summary>Visible RGB32 aperture and display orientation, independent of coded padding.
/// Rotation undoes the counter-clockwise content rotation reported by Media Foundation.</summary>
internal sealed class VideoFrameLayout
{
    internal const int MaximumDimension=8192,MaximumPixels=33_177_600;
    const int MaximumRowPadding=4096;
    readonly int bufferWidth,bufferHeight,x,y,width,height,rotation;
    public int Width=>rotation is 90 or 270?height:width;
    public int Height=>rotation is 90 or 270?width:height;

    public VideoFrameLayout(int bufferWidth,int bufferHeight,int x,int y,int width,int height,int rotation)
    {
        ValidateDimensions(bufferWidth,bufferHeight);
        if(x<0||y<0||width<=0||height<=0||(long)x+width>bufferWidth||(long)y+height>bufferHeight)
            throw new InvalidDataException("Video display aperture is outside its decoded frame.");
        if(rotation is not (0 or 90 or 180 or 270))throw new InvalidDataException("Unsupported video rotation metadata.");
        this.bufferWidth=bufferWidth;this.bufferHeight=bufferHeight;this.x=x;this.y=y;
        this.width=width;this.height=height;this.rotation=rotation;
    }

    public byte[] CopyRgba(IntPtr data,int length,int stride,CancellationToken token)
    {
        ValidateStride(bufferWidth,bufferHeight,stride);
        var pitch=Math.Abs((long)stride);
        if(data==IntPtr.Zero||pitch*bufferHeight>length)
            throw new InvalidDataException("Decoded buffer is shorter than its frame dimensions or stride.");
        var result=new byte[checked(Width*Height*4)];
        var row=rotation==0?null:new byte[checked(width*4)];
        for(var sourceY=0;sourceY<height;sourceY++)
        {
            token.ThrowIfCancellationRequested();
            var bufferY=stride<0?bufferHeight-1-y-sourceY:y+sourceY;
            var source=IntPtr.Add(data,checked((int)(bufferY*pitch+x*4L)));
            if(rotation==0)Marshal.Copy(source,result,sourceY*width*4,width*4);
            else
            {
                Marshal.Copy(source,row!,0,width*4);
                for(var sourceX=0;sourceX<width;sourceX++)
                {
                    var destination=rotation switch
                    {
                        90=>(sourceX*Width+height-1-sourceY)*4,
                        180=>((height-1-sourceY)*Width+width-1-sourceX)*4,
                        _=>((width-1-sourceX)*Width+sourceY)*4
                    };
                    var offset=sourceX*4;
                    result[destination]=row![offset];result[destination+1]=row[offset+1];
                    result[destination+2]=row[offset+2];
                }
            }
        }
        for(var i=0;i<result.Length;i+=4){(result[i],result[i+2])=(result[i+2],result[i]);result[i+3]=255;}
        return result;
    }

    internal static void ValidateDimensions(int width,int height)
    {
        if(width<=0||height<=0||width>MaximumDimension||height>MaximumDimension||(long)width*height>MaximumPixels)
            throw new InvalidDataException("Video exceeds the decode limit of 8192 pixels per side or 33,177,600 pixels per frame.");
    }

    internal static void ValidateStride(int width,int height,int stride)
    {
        ValidateDimensions(width,height);
        var pitch=Math.Abs((long)stride);
        // Allow a page of native row alignment, but reject malicious multi-gigabyte pitches.
        if(pitch<width*4L||pitch>width*4L+MaximumRowPadding)
            throw new InvalidDataException("Video row stride is outside the decoded width and alignment limit.");
    }
}
