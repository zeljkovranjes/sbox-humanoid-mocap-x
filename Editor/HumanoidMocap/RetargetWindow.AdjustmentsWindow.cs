using System.Collections.Generic;
using System.Linq;
using Editor;
using Sandbox;

namespace HumanoidMocapX.EditorTools;

public sealed partial class RetargetWindow
{
    sealed record AdjustmentPage(string Category,string Icon,string Title,Widget Page,bool Listed);
    readonly List<AdjustmentPage> _adjustmentPages=new();
    readonly List<AdjustmentNode> _adjustmentNodes=new();
    TreeView _adjustmentTree;
    const float AdjustmentSidebarWidth=200;

    /// <summary>Adds a page to Mocap adjustments under <paramref name="category"/>. Unlisted pages are
    /// built (their controls stay in use) but get no sidebar entry.</summary>
    Widget AddAdjustmentPage(string category,string icon,string title,bool listed=true)
    {
        var page=_advancedPanel.Layout.Add(new Widget(this));page.Layout=Layout.Column();page.Layout.Spacing=8;
        _adjustmentPages.Add(new(category,icon,title,page,listed));
        return page;
    }

    static string CategoryIcon(string category)=>category switch{"Processing"=>"settings","Capture"=>"movie_filter","Character"=>"accessibility_new",_=>"auto_fix_high"};

    /// <summary>Adds a host page to Mocap adjustments and returns its body (a column with the
    /// page title above it). Call before the window is first opened; pages added later never show.</summary>
    public Widget AddSettingsPage(string category,string icon,string title)
    {
        var card=AddAdjustmentPage(category,icon,title).Layout.Add(new MocapCard(this){Flat=true});
        card.Header(icon,title);
        return card;
    }

    /// <summary>The host's setup gate. While <paramref name="reason"/> is set, the drop area is grayed
    /// out and shows it, and the settings button carries a red dot; null lifts the gate.</summary>
    public void SetSetupRequired(string reason)
    {
        if(_welcome is VideoDropArea drop)drop.SetLocked(reason);
        _settingsButton.Badge=reason is not null;
        _settingsButton.ToolTip=reason is null?"Mocap adjustments":"Mocap adjustments: finish setup first";
    }

    /// <summary>Mocap adjustments: a category tree on the left and the selected pages on the right,
    /// like the editor's Project Settings, in the workspace's gray rather than black.</summary>
    internal void ShowAdjustments()=>ShowAdjustments(null);

    /// <summary>Opens Mocap adjustments, or brings it forward, on the page titled
    /// <paramref name="title"/>; null or an unknown title keeps the current or first page.</summary>
    public void ShowAdjustments(string title)
    {
        if(_adjustments.IsValid()){SelectAdjustmentPage(title);_adjustments.Show();_adjustments.Window.Raise();return;}
        _adjustments=new Dialog(this);_adjustments.Window.Title="Mocap adjustments";_adjustments.Window.SetWindowIcon("videocam");
        _adjustments.OnPaintOverride=()=>{Paint.ClearPen();Paint.SetBrush(Theme.WidgetBackground);Paint.DrawRect(_adjustments.LocalRect);return true;};
        _adjustments.Layout=Layout.Row();
        var tree=_adjustmentTree=_adjustments.Layout.Add(new TreeView(_adjustments){FixedWidth=AdjustmentSidebarWidth});
        tree.Margin=tree.Margin with{Top=4,Bottom=4};
        tree.OnPaintOverride=()=>
        {
            Paint.ClearPen();Paint.SetBrush(Color.Lerp(Theme.WidgetBackground,Color.Black,.12f));Paint.DrawRect(tree.LocalRect);
            Paint.SetPen(MocapButton.Edge);Paint.DrawLine(tree.LocalRect.TopRight,tree.LocalRect.BottomRight);
            return false;
        };
        var scroll=_adjustments.Layout.Add(new ScrollArea(_adjustments),1);
        _advancedPanel.Parent=scroll;scroll.Canvas=_advancedPanel;_advancedPanel.Visible=true;
        // Right margin keeps the content clear of the vertical scrollbar, as MappingEditor does.
        _advancedPanel.Layout.Margin=new Sandbox.UI.Margin(12,10,16,10);_advancedPanel.Layout.Spacing=8;
        // Pages start at the top instead of floating in the middle of the window.
        _advancedPanel.Layout.AddStretchCell();
        AdjustmentNode first=null;
        foreach(var group in _adjustmentPages.Where(page=>page.Listed).GroupBy(page=>page.Category))
        {
            var category=new AdjustmentNode(group.Key,CategoryIcon(group.Key),group.Select(page=>page.Page).ToArray(),true);
            tree.AddItem(category);_adjustmentNodes.Add(category);
            foreach(var page in group)
            {
                var node=new AdjustmentNode(page.Title,page.Icon,new[]{page.Page},false);
                first??=node;category.AddItem(node);_adjustmentNodes.Add(node);
            }
            tree.Open(category);
        }
        foreach(var page in _adjustmentPages)page.Page.Visible=false;
        // The tree reports a row by its Value, which is the row's page list.
        tree.ItemSelected+=item=>{if(item is Widget[] pages)ShowAdjustmentPages(pages);};
        if(first is not null){tree.SelectItem(first);ShowAdjustmentPages(first.Pages);}
        SelectAdjustmentPage(title);
        // Preserve the settings widgets when the user closes this separate window.
        _adjustments.Window.DeleteOnClose=false;
        _adjustments.Window.MinimumSize=new Vector2(860,480);_adjustments.Window.Size=new Vector2(940,600);_adjustments.Show();
    }

    void SelectAdjustmentPage(string title)
    {
        if(title is null||_adjustmentNodes.FirstOrDefault(node=>node.Name==title) is not { } node)return;
        _adjustmentTree.SelectItem(node);ShowAdjustmentPages(node.Pages);
    }

    /// <summary>Shows the given pages, in their built order, and hides the rest.</summary>
    void ShowAdjustmentPages(Widget[] shown){foreach(var page in _adjustmentPages)page.Page.Visible=shown.Contains(page.Page);}

    /// <summary>A sidebar row: a bold category or an indented page, with a lighter fill on hover and
    /// the icon buttons' green when selected.</summary>
    sealed class AdjustmentNode : TreeNode
    {
        const float IconSize=16,Inset=6,Gap=8;
        readonly string icon;
        readonly bool isCategory;
        internal readonly Widget[] Pages;
        public AdjustmentNode(string title,string icon,Widget[] pages,bool isCategory)
        {Name=title;this.icon=icon;Pages=pages;this.isCategory=isCategory;Height=isCategory?26:24;
            // The tree tracks rows by Value; each row's own page list keeps them distinct.
            Value=pages;}
        public override void OnPaint(VirtualWidget item)
        {
            Paint.Antialiasing=true;
            var row=item.Rect.Shrink(2,1,6,1);
            if(item.Selected||item.Hovered)
            {
                Paint.ClearPen();Paint.SetBrush(item.Selected?Theme.Green.WithAlpha(.15f):MocapButton.Fill.Lighten(.3f));
                Paint.DrawRect(row,4);
            }
            Paint.SetPen(item.Selected?Theme.Green:isCategory?Theme.Text:Theme.TextLight);
            var iconRect=Paint.DrawIcon(new Rect(row.Left+Inset,row.Top,IconSize,row.Height),icon,IconSize,TextFlag.LeftCenter);
            var text=row;text.Left=iconRect.Right+Gap;
            Paint.SetPen(Theme.Text);
            if(isCategory)Paint.SetDefaultFont(weight:600);else Paint.SetDefaultFont();
            Paint.DrawText(text,Name,TextFlag.LeftCenter);
        }
    }
}
