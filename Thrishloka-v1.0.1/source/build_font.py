# Copyright (c) 2026, Sneha.
# SPDX-License-Identifier: OFL-1.1
# Licensed under the SIL Open Font License 1.1; see ../OFL.txt.
from pathlib import Path
import math, json, unicodedata, copy
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0
from fontTools.ttLib.tables.O_S_2f_2 import Panose
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools import agl

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'work'
OUT=ROOT/'build'
IMG=ROOT/'sources'
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
UPM=1000
GLYPHS={}; METRICS={}; CMAP={}; SOURCES={}

def readmask(name):
    a=np.asarray(Image.open(IMG/name).convert('RGB'))
    return (a.min(axis=2)<120).astype(np.uint8)

def runs(values):
    nz=np.flatnonzero(values)
    if not len(nz): return []
    splits=np.flatnonzero(np.diff(nz)>1)+1
    return [(int(g[0]),int(g[-1]+1)) for g in np.split(nz,splits)]

def sheet(mask, bands, strings):
    result={}
    for (ya,yb),text in zip(bands,strings):
        row=mask[ya:yb]
        xr=runs(row.sum(axis=0)>1)
        assert len(xr)==len(text),(text,xr)
        for ch,(xa,xb) in zip(text,xr):
            yy=np.flatnonzero(row[:,xa:xb].sum(axis=1))
            result[ch]=(row[yy[0]:yy[-1]+1,xa:xb].copy(),(xa,ya+int(yy[0]),xb-xa,int(yy[-1]-yy[0]+1)))
    return result

def unit(v):
    n=np.linalg.norm(v)
    return v/n if n>1e-10 else np.array([1.,0.])

def fit(points, t0, t1, tol=0.65, depth=0):
    # Least-squares cubic Bezier fitting with recursive error subdivision.
    if len(points)==2:
        d=np.linalg.norm(points[1]-points[0])/3
        return [(points[0],points[0]+t0*d,points[-1]+t1*d,points[-1])]
    dist=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    u=dist/max(dist[-1],1e-9); v=1-u
    b0=v**3; b1=3*u*v*v; b2=3*u*u*v; b3=u**3
    a0=b1[:,None]*t0; a1=b2[:,None]*t1
    rhs=points-(b0+b1)[:,None]*points[0]-(b2+b3)[:,None]*points[-1]
    mat=np.array([[np.sum(a0*a0),np.sum(a0*a1)],[np.sum(a0*a1),np.sum(a1*a1)]])
    vec=np.array([np.sum(a0*rhs),np.sum(a1*rhs)])
    try: alpha=np.linalg.solve(mat,vec)
    except np.linalg.LinAlgError: alpha=np.array([-1.,-1.])
    d=np.linalg.norm(points[-1]-points[0])
    if min(alpha)<1e-5 or max(alpha)>max(d*4,1): alpha=np.array([d/3,d/3])
    bez=(points[0],points[0]+t0*alpha[0],points[-1]+t1*alpha[1],points[-1])
    pred=b0[:,None]*bez[0]+b1[:,None]*bez[1]+b2[:,None]*bez[2]+b3[:,None]*bez[3]
    error=np.sum((pred-points)**2,axis=1); k=int(np.argmax(error))
    if error[k]<=tol*tol or depth>25: return [bez]
    k=max(1,min(k,len(points)-2))
    tc=unit(points[k-1]-points[k+1])
    return fit(points[:k+1],t0,tc,tol,depth+1)+fit(points[k:],-tc,t1,tol,depth+1)

def trace(mask, sx, sy, baseline, lsb, tol=.6):
    padded=np.pad(mask,3)
    contours,hier=cv2.findContours(padded,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    pen=TTGlyphPen(None); curve=Cu2QuPen(pen,1.0,reverse_direction=False)
    def tx(p): return (float((p[0]-3)*sx+lsb),float((baseline-(p[1]-3))*sy))
    for i,c in enumerate(contours):
        if abs(cv2.contourArea(c))<4: continue
        pts=c[:,0,:].astype(float)
        # Native outer contours become counterclockwise after y-flip; reverse
        # to TrueType's clockwise outer contour convention, preserving holes.
        pts=pts[::-1]
        # Split closed outlines at four approximately equally distant points.
        # The fitted pieces share endpoints and follow the source contour.
        n=len(pts); cuts=[0,n//4,n//2,3*n//4,n]
        curve.moveTo(tx(pts[0]))
        closed=np.vstack([pts,pts[0]])
        for a,b in zip(cuts,cuts[1:]):
            sub=closed[a:b+1]
            t0=unit(sub[min(3,len(sub)-1)]-sub[0])
            t1=unit(sub[max(0,len(sub)-4)]-sub[-1])
            for bez in fit(sub,t0,t1,tol):
                curve.curveTo(tx(bez[1]),tx(bez[2]),tx(bez[3]))
        curve.closePath()
    return pen.glyph()

def name(ch): return agl.UV2AGL.get(ord(ch),f'uni{ord(ch):04X}')

def register(ch,g,advance,lsb=40,source='constructed'):
    n=name(ch);GLYPHS[n]=g;METRICS[n]=(int(round(advance)),int(round(lsb)));CMAP[ord(ch)]=n;SOURCES[ch]=source

def frommask(ch,mask,sy,baseline=None,sx=None,lsb=35,rsb=35,advance=None,source='reference trace'):
    # Remove fragments of adjacent letters caught inside a tight source crop.
    count,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    if count>1:
        areas=stats[1:,4]; keep=np.flatnonzero(areas>=max(areas)*.06)+1
        mask=np.isin(labels,keep).astype(np.uint8)
    if sx is None:sx=sy
    if baseline is None:baseline=mask.shape[0]-1
    w=(mask.shape[1]-1)*sx
    if advance is not None:lsb=(advance-w)/2
    g=trace(mask,sx,sy,baseline,lsb)
    register(ch,g,advance or (w+lsb+rsb),lsb,source)

upper=sheet(readmask('uppercase-source.png'),[(0,280),(280,530),(530,810),(810,1024)],['ABCDEFG','HIJKLMN','OPQRSTU','VWXYZ'])
lower=sheet(readmask('lowercase-source.png'),[(0,325),(325,600),(600,860),(860,1060)],['abcdefg','hijklmn','opqrstu','vwxyz'])

for ch,(m,box) in upper.items():
    h=m.shape[0]-1
    if ch=='Q':sy=720/185;baseline=185
    elif ch in 'CGOSU':sy=720/h;baseline=h-10/sy
    elif ch=='J':sy=710/h;baseline=h-10/sy
    else:sy=700/h;baseline=h
    frommask(ch,m,sy,baseline,sx=sy,lsb=32,rsb=32,source='extended uppercase sheet')

for ch,(m,box) in lower.items():
    x,y,w,h=box
    # Align row x-heights first; ascenders and descenders retain their anatomy.
    if ch in 'abcdefg':base=260;xh=145
    elif ch in 'hijklmn':base=547;xh=137
    elif ch in 'opqrstu':base=800;xh=145
    else:base=1021;xh=127
    sy=500/xh
    sx=sy*.94
    baseline=base-y
    if ch in 'bdhkl':
        sy=700/(baseline or 1)
    if ch in 'gpqy':
        # The generated sheet has shortened descenders. Set a useful text depth.
        desired=190
        # Piecewise extend only the portion below the baseline before tracing.
        bottom=max(1,m.shape[0]-1-baseline)
        want=max(1,round(desired/sy))
        top=m[:baseline+1]
        tail=cv2.resize(m[baseline+1:],(w,want),interpolation=cv2.INTER_NEAREST)
        m=np.vstack([top,tail])
    if ch=='j':
        sy=700/(547-353);baseline=547-y
    frommask(ch,m,sy,baseline,sx=sx,lsb=32,rsb=32,source='extended lowercase sheet')

# A dotless i is needed for properly constructed accented lowercase i forms.
im,ibox=lower['i']; count,labels,stats,centers=cv2.connectedComponentsWithStats(im)
largest=1+np.argmax(stats[1:,4]); im=(labels==largest).astype(np.uint8)
frommask('ı',im,500/137,547-ibox[1],sx=(500/137)*.94,lsb=32,rsb=32,source='dotless form of extended lowercase i')

# Anchor the most recognizable letters directly to the approved Regular image.
approved=readmask('approved-regular-reference.png')
anchors={'C':(30,25,181,233,256),'u':(221,95,146,164,256),
 'r':(380,92,91,164,256),'e':(571,92,130,168,256),
 'n':(710,92,147,164,256),'t':(854,40,105,219,256),
 'o':(1016,92,145,167,256),'s':(1609,92,114,168,256),
 'a':(35,373,154,171,539),'d':(364,310,154,234,539),
 'm':(604,374,249,165,539)}
for ch,(x,y,w,h,b) in anchors.items():
    frommask(ch,approved[y:y+h,x:x+w],700/231,b-y,lsb=32,rsb=32,source='approved regular preview')

samples=[('A',26,694,109,103),('E',251,694,77,103),('G',444,691,99,108),('R',672,694,87,104),('&',861,691,100,108)]
digits=[(1005,693,77,107),(1096,694,51,103),(1159,693,72,105),(1246,693,69,107),(1331,694,77,103),(1424,695,69,105),(1511,693,73,107),(1594,695,68,103),(1674,693,71,107),(1761,693,73,107)]
for ch,x,y,w,h in samples:
    frommask(ch,approved[y:y+h,x:x+w],700/(h-1),lsb=32,rsb=32,source='approved regular preview')
for ch,(x,y,w,h) in zip('0123456789',digits):
    frommask(ch,approved[y:y+h,x:x+w],700/(h-1),advance=580,source='approved regular preview; tabular figures')

# Native outline construction for punctuation and common symbols.
def pathglyph(draw):
    p=TTGlyphPen(None);q=Cu2QuPen(p,1.,reverse_direction=False);draw(q);return p.glyph()
def polygon(p,points):
    p.moveTo(points[0])
    for xy in points[1:]:p.lineTo(xy)
    p.closePath()
def rect(p,x,y,w,h):polygon(p,[(x,y),(x,y+h),(x+w,y+h),(x+w,y)])
def ellipse(p,cx,cy,rx,ry,reverse=False):
    k=.55228475
    pts=[(cx+rx,cy),(cx+rx,cy+k*ry),(cx+k*rx,cy+ry),(cx,cy+ry),(cx-k*rx,cy+ry),(cx-rx,cy+k*ry),(cx-rx,cy),(cx-rx,cy-k*ry),(cx-k*rx,cy-ry),(cx,cy-ry),(cx+k*rx,cy-ry),(cx+rx,cy-k*ry)]
    # Clockwise outer contours.
    p.moveTo(pts[0])
    if reverse:
        p.curveTo(pts[1],pts[2],pts[3]);p.curveTo(pts[4],pts[5],pts[6]);p.curveTo(pts[7],pts[8],pts[9]);p.curveTo(pts[10],pts[11],pts[0])
    else:
        p.curveTo(pts[11],pts[10],pts[9]);p.curveTo(pts[8],pts[7],pts[6]);p.curveTo(pts[5],pts[4],pts[3]);p.curveTo(pts[2],pts[1],pts[0])
    p.closePath()
def line(p,x1,y1,x2,y2,width=48):
    dx=x2-x1;dy=y2-y1;n=math.hypot(dx,dy);ox=-dy/n*width/2;oy=dx/n*width/2
    polygon(p,[(x1+ox,y1+oy),(x2+ox,y2+oy),(x2-ox,y2-oy),(x1-ox,y1-oy)])
def symbol(ch,advance,draw):register(ch,pathglyph(draw),advance,35)
def compound(ch,parts,advance):
    p=TTGlyphPen(GLYPHS)
    for base,tr in parts:p.addComponent(name(base),tr)
    register(ch,p.glyph(),advance,32)
def ident(dx=0,dy=0):return (1,0,0,1,dx,dy)
def quote(p,x,y):
    p.moveTo((x,y+85));p.curveTo((x+52,y+90),(x+58,y+28),(x+12,y-36));p.lineTo((x-8,y-22));p.curveTo((x+12,y+9),(x+17,y+32),(x,y+31));p.curveTo((x-39,y+25),(x-39,y+83),(x,y+85));p.closePath()

symbol(' ',280,lambda p:None)
symbol('.',250,lambda p:ellipse(p,125,40,40,40))
symbol(',',250,lambda p:quote(p,125,-25))
symbol(':',250,lambda p:[ellipse(p,125,40,37,37),ellipse(p,125,370,37,37)])
symbol(';',250,lambda p:[quote(p,125,-25),ellipse(p,125,370,37,37)])
symbol('!',280,lambda p:[polygon(p,[(105,700),(175,700),(160,200),(120,200)]),ellipse(p,140,40,40,40)])
def question(p):
    p.moveTo((90,557));p.curveTo((70,720),(410,790),(430,580));p.curveTo((441,470),(260,415),(260,255));p.lineTo((213,255));p.curveTo((195,435),(355,479),(356,582));p.curveTo((355,708),(162,674),(150,567));p.curveTo((143,525),(96,520),(90,557));p.closePath();ellipse(p,238,40,40,40)
symbol('?',500,question)
symbol('-',360,lambda p:rect(p,55,260,250,48))
symbol('_',600,lambda p:rect(p,25,-120,550,45))
symbol('+',640,lambda p:[rect(p,75,260,490,48),rect(p,296,40,48,490)])
symbol('=',640,lambda p:[rect(p,75,170,490,48),rect(p,75,355,490,48)])
symbol('/',480,lambda p:line(p,65,-90,415,755,48))
symbol('\\',480,lambda p:line(p,65,755,415,-90,48))
symbol('|',260,lambda p:rect(p,106,-180,48,960))
symbol('<',600,lambda p:[line(p,500,535,100,285),line(p,100,285,500,35)])
symbol('>',600,lambda p:[line(p,100,535,500,285),line(p,500,285,100,35)])
symbol("'",230,lambda p:quote(p,100,610))
symbol('"',410,lambda p:[quote(p,105,610),quote(p,285,610)])
symbol('`',350,lambda p:polygon(p,[(80,765),(145,785),(260,630),(223,610)]))
symbol('^',550,lambda p:[line(p,75,575,275,745,42),line(p,275,745,475,575,42)])
def tilde(p):
    p.moveTo((70,600));p.curveTo((200,800),(350,550),(480,715));p.lineTo((490,655));p.curveTo((345,480),(210,740),(70,545));p.closePath()
symbol('~',560,tilde)
symbol('*',500,lambda p:[line(p,250,420,250,720,35),line(p,107,524,393,616,35),line(p,162,691,338,449,35)])
symbol('#',660,lambda p:[line(p,200,0,300,700,45),line(p,400,0,500,700,45),rect(p,80,220,500,48),rect(p,110,465,500,48)])
def paren(p):
    p.moveTo((285,780));p.curveTo((40,540),(40,80),(285,-160));p.lineTo((310,-132));p.curveTo((125,145),(125,475),(310,752));p.closePath()
symbol('(',350,paren)
compound(')',[('(',(-1,0,0,1,350,0))],350)
symbol('[',350,lambda p:[rect(p,110,-150,45,920),rect(p,110,725,190,45),rect(p,110,-150,190,45)])
compound(']',[('[',(-1,0,0,1,350,0))],350)
def brace(p):
    p.moveTo((330,780));p.curveTo((110,790),(145,620),(145,480));p.curveTo((145,360),(130,330),(60,310));p.lineTo((60,270));p.curveTo((130,250),(145,220),(145,100));p.curveTo((145,-40),(110,-210),(330,-200));p.lineTo((330,-157));p.curveTo((200,-160),(210,-50),(210,90));p.curveTo((210,210),(185,265),(115,290));p.curveTo((185,315),(210,370),(210,490));p.curveTo((210,630),(200,740),(330,737));p.closePath()
symbol('{',400,brace)
compound('}',[('{',(-1,0,0,1,400,0))],400)
def percent(p):
    ellipse(p,170,540,105,155);ellipse(p,170,540,62,112,True)
    ellipse(p,490,150,105,155);ellipse(p,490,150,62,112,True)
    line(p,135,0,525,700,38)
symbol('%',660,percent)
def atsign(p):
    p.moveTo((670,120));p.curveTo((510,-70),(85,-30),(70,340));p.curveTo((55,780),(790,840),(790,370));p.curveTo((790,115),(550,60),(515,225));p.lineTo((560,540));p.lineTo((615,540));p.lineTo((570,240));p.curveTo((560,110),(735,190),(735,370));p.curveTo((735,730),(110,730),(120,340));p.curveTo((130,35),(505,-5),(642,152));p.closePath()
    ellipse(p,395,365,155,200);ellipse(p,395,365,95,150,True)
symbol('@',870,atsign)
symbol('$',METRICS[name('S')][0],lambda p:None)
# Composite currency shapes retain the letter family's contours.
def add_to(ch,base,draw,advance=None):
    class Wrapper:
        def __init__(self,g):self.g=g
        def draw(self,p):self.g.draw(p,GLYPHS)
    rec=DecomposingRecordingPen({n:Wrapper(g) for n,g in GLYPHS.items()})
    GLYPHS[name(base)].draw(rec,GLYPHS)
    p=TTGlyphPen(None);rec.replay(p)
    q=Cu2QuPen(p,1.);draw(q)
    register(ch,p.glyph(),advance or METRICS[name(base)][0],32)
add_to('$','S',lambda p:rect(p,METRICS[name('S')][0]/2-15,-80,30,860))
add_to('¢','c',lambda p:line(p,230,-80,320,650,32))
add_to('€','C',lambda p:[rect(p,30,280,340,40),rect(p,30,410,340,40)])
add_to('£','L',lambda p:rect(p,25,350,350,45))
add_to('¥','Y',lambda p:[rect(p,125,200,370,40),rect(p,125,315,370,40)])
def rupee(p):
    rect(p,70,650,460,45);rect(p,70,530,460,40)
    p.moveTo((240,670));p.curveTo((530,650),(530,355),(165,355));p.lineTo((470,0));p.lineTo((380,0));p.lineTo((75,360));p.lineTo((75,400));p.curveTo((460,370),(440,630),(240,625));p.closePath()
symbol('₹',600,rupee)

symbol('–',600,lambda p:rect(p,35,260,530,48))
symbol('—',1000,lambda p:rect(p,35,260,930,48))
compound('…',[('.',ident()),('.',ident(250)),('.',ident(500))],750)
compound('’',[("'",ident())],230)
compound('‘',[("'",(-1,0,0,-1,230,1300))],230)
compound('”',[('"',ident())],410)
compound('“',[('"',(-1,0,0,-1,410,1300))],410)
symbol('•',400,lambda p:ellipse(p,200,280,80,80))
symbol('·',250,lambda p:ellipse(p,125,280,35,35))
symbol('°',400,lambda p:[ellipse(p,200,560,120,130),ellipse(p,200,560,74,84,True)])
symbol('×',640,lambda p:[line(p,115,70,525,500),line(p,115,500,525,70)])
symbol('÷',640,lambda p:[rect(p,75,260,490,48),ellipse(p,320,100,35,35),ellipse(p,320,470,35,35)])
compound('±',[('+',ident(0,110)),('-', (1.96,0,0,1,-32,-260))],640)
compound('−',[('–',ident())],600)
symbol('†',500,lambda p:[rect(p,225,-110,50,810),rect(p,50,430,400,45)])
symbol('‡',500,lambda p:[rect(p,225,-110,50,810),rect(p,50,430,400,45),rect(p,50,140,400,45)])
compound('¡',[('!',(-1,0,0,-1,280,520))],280)
compound('¿',[('?',(-1,0,0,-1,500,520))],500)
compound('«',[('<',(.5,0,0,.65,20,80)),('<',(.5,0,0,.65,260,80))],590)
compound('»',[('>',(.5,0,0,.65,20,80)),('>',(.5,0,0,.65,260,80))],590)
symbol('¬',640,lambda p:[rect(p,75,330,490,48),rect(p,517,120,48,250)])
symbol('¦',260,lambda p:[rect(p,106,-180,48,410),rect(p,106,370,48,410)])
symbol('¤',660,lambda p:[ellipse(p,330,350,185,185),ellipse(p,330,350,135,135,True),line(p,100,120,205,225,40),line(p,455,475,560,580,40),line(p,100,580,205,475,40),line(p,455,225,560,120,40)])

# Extra symbols use outline components, not substitutions from other fonts.
for ch,letter in [('©','C'),('®','R')]:
    ring=pathglyph(lambda p:[ellipse(p,400,350,350,360),ellipse(p,400,350,305,315,True)])
    gn=f'ring{ord(ch)}';GLYPHS[gn]=ring;METRICS[gn]=(800,50)
    p=TTGlyphPen(GLYPHS);p.addComponent(gn,ident());p.addComponent(name(letter),(.58,0,0,.58,215,145));register(ch,p.glyph(),800,50)
compound('™',[('T',(.48,0,0,.48,0,370)),('M',(.48,0,0,.48,310,370))],730)
compound('§',[('S',(.75,0,0,.72,40,170)),('S',(.75,0,0,.72,40,-170))],550)
compound('µ',[('u',ident()),('|',(.7,0,0,.26,-40,-160))],METRICS[name('u')][0])
compound('¶',[('P',(-1,0,0,1,620,0)),('|',(1,0,0,.85,240,-60))],650)
compound('ª',[('a',(.6,0,0,.6,0,400))],340)
compound('º',[('o',(.6,0,0,.6,0,400))],340)
for ch,d in [('¹','1'),('²','2'),('³','3')]:compound(ch,[(d,(.6,0,0,.6,0,400))],348)
for ch,a,b in [('¼','1','4'),('½','1','2'),('¾','3','4')]:
    compound(ch,[(a,(.52,0,0,.52,0,360)),('/',(.65,0,0,.8,240,40)),(b,(.52,0,0,.52,500,0))],820)

# Accents and Latin-1 letters. Marks are designed in this font's own outlines.
marks={
 '\u0301': lambda p:polygon(p,[(170,610),(285,785),(350,765),(210,590)]),
 '\u0300': lambda p:polygon(p,[(150,765),(215,785),(330,610),(290,590)]),
 '\u0302': lambda p:[line(p,100,610,250,755,38),line(p,250,755,400,610,38)],
 '\u0303': tilde,
 '\u0308': lambda p:[ellipse(p,145,670,33,33),ellipse(p,355,670,33,33)],
 '\u030A': lambda p:[ellipse(p,250,675,85,85),ellipse(p,250,675,48,48,True)],
 '\u0327': lambda p:[polygon(p,[(245,0),(215,-80),(290,-100),(270,-160),(180,-175),(180,-210),(325,-195),(350,-75),(265,-50),(285,0)])]
}
for ch,draw in marks.items():symbol(ch,0,draw)
for cp in range(0xC0,0x100):
    ch=chr(cp);decomp=unicodedata.normalize('NFD',ch)
    if len(decomp)==2 and decomp[0] in SOURCES and decomp[1] in marks:
        base,accent=decomp;adv=METRICS[name(base)][0]
        dy=190 if base.isupper() and accent!='\u0327' else 0
        if base=='i':base='ı'
        compound(ch,[(base,ident()),(accent,ident((adv-500)/2,dy))],adv)
compound('\u00A0',[(' ',ident())],280)
compound('\u00AD',[('-',ident())],360)
compound('´',[('\u0301',ident())],500)
compound('¨',[('\u0308',ident())],500)
compound('¸',[('\u0327',ident())],500)
symbol('¯',500,lambda p:rect(p,90,650,320,40))
compound('Ø',[('O',ident()),('/',(1.25,0,0,.9,30,30))],METRICS[name('O')][0])
compound('ø',[('o',ident()),('/',(.95,0,0,.65,0,20))],METRICS[name('o')][0])
compound('Æ',[('A',(.8,0,0,1,0,0)),('E',(.85,0,0,1,490,0))],1000)
compound('æ',[('a',(.8,0,0,1,0,0)),('e',(.9,0,0,1,330,0))],760)
compound('Œ',[('O',(.85,0,0,1,0,0)),('E',(.85,0,0,1,480,0))],1000)
compound('œ',[('o',(.85,0,0,1,0,0)),('e',(.9,0,0,1,340,0))],760)
add_to('Ð','D',lambda p:rect(p,0,330,300,42))
compound('ð',[('d',ident())],METRICS[name('d')][0])
add_to('ð','ð',lambda p:line(p,200,620,440,760,35))
compound('Þ',[('P',(1,0,0,.73,0,50)),('|',(.7,0,0,.73,-12,130))],METRICS[name('P')][0])
compound('þ',[('p',ident()),('|',(.7,0,0,.65,-12,180))],METRICS[name('p')][0])
def eszett(p):
    p.moveTo((70,0));p.lineTo((70,510));p.curveTo((70,790),(470,790),(470,560));p.curveTo((470,470),(410,425),(350,400));p.curveTo((650,250),(550,-55),(270,0));p.lineTo((270,48));p.curveTo((485,0),(515,225),(290,355));p.lineTo((290,420));p.curveTo((420,440),(440,650),(290,670));p.curveTo((130,680),(145,550),(145,490));p.lineTo((145,0));p.closePath()
symbol('ß',590,eszett)
compound('ẞ',[('ß',(1.13,0,0,1.02,0,0))],670)

# Nonspacing marks are available for decomposed Latin text.
for cp in [0x2000,0x2001,0x2002,0x2003,0x2004,0x2005,0x2006,0x2007,0x2008,0x2009,0x200A,0x202F]:
    width={0x2000:500,0x2001:1000,0x2002:500,0x2003:1000,0x2004:333,0x2005:250,0x2006:167,0x2007:580,0x2008:250,0x2009:200,0x200A:100,0x202F:200}[cp]
    symbol(chr(cp),width,lambda p:None)
for cp in [0x200B,0x200C,0x200D,0x2060,0xFEFF]:symbol(chr(cp),0,lambda p:None)

# Build a proper Unicode TrueType/OpenType font.
GLYPHS['.notdef']=pathglyph(lambda p:[rect(p,70,0,480,700),polygon(p,[(120,50),(500,50),(500,650),(120,650)])])
METRICS['.notdef']=(620,70)
order=['.notdef']+[n for n in GLYPHS if n!='.notdef']
fb=FontBuilder(UPM,isTTF=True)
fb.setupGlyphOrder(order);fb.setupCharacterMap(CMAP)
fb.setupGlyf(GLYPHS)
# Recompute left sidebearings from actual geometry, including composites.
for n,g in GLYPHS.items():
    if g.numberOfContours or g.isComposite():
        g.recalcBounds(fb.font['glyf']);METRICS[n]=(METRICS[n][0],getattr(g,'xMin',0))
    else:METRICS[n]=(METRICS[n][0],0)
fb.setupHorizontalMetrics(METRICS)
fb.setupHorizontalHeader(ascent=1050,descent=-280,lineGap=0)
license_text=(ROOT.parent/'OFL.txt').read_text(encoding='utf-8')
fb.setupNameTable({
 'copyright':'Copyright (c) 2026, Sneha.',
 'familyName':'Thrishloka','styleName':'Regular',
 'typographicFamily':'Thrishloka','typographicSubfamily':'Regular',
 'uniqueFontIdentifier':'Thrishloka-Regular-1.001-OFL-20261112',
 'fullName':'Thrishloka Regular','psName':'Thrishloka-Regular',
 'version':'Version 1.001','manufacturer':'Sneha',
 'designer':'Sneha; AI-assisted design and font engineering',
 'description':'Thrishloka Regular. Initial public preview. Custom semi-serif with slightly condensed proportions, soft flares, and open counters. See SOURCE-NOTES.txt for provenance.',
 'licenseDescription':license_text,'licenseInfoURL':'https://openfontlicense.org'
})
fb.setupOS2(sTypoAscender=1050,sTypoDescender=-280,sTypoLineGap=0,usWinAscent=1050,usWinDescent=280,sxHeight=500,sCapHeight=700,usWeightClass=400,usWidthClass=4,fsType=0,fsSelection=0xC0)
fb.setupPost();fb.setupMaxp()
fb.font['head'].macStyle=0
fb.font['OS/2'].panose=Panose()
for attr in ['bFamilyType','bSerifStyle','bWeight','bProportion','bContrast','bStrokeVariation','bArmStyle','bLetterForm','bMidline','bXHeight']:
    setattr(fb.font['OS/2'].panose,attr,0)
fb.font['OS/2'].version=4
fb.font['OS/2'].panose.bFamilyType=2;fb.font['OS/2'].panose.bSerifStyle=7
fb.font['OS/2'].panose.bWeight=5;fb.font['OS/2'].panose.bProportion=6
fb.font['head'].fontRevision=1.001

# Kern visibly sensitive pairs; use GPOS plus legacy kern for older applications.
kern={}
for left,rights,value in [('A','VWYT',-65),('V','Aaeou',-65),('W','Aaeou',-45),('Y','Aaeou',-75),('T','aeou',-55),('F','ao',-35),('P','ao',-35),('L','TVWY',-55),('r','.,',-30),('v','aeo',-20),('w','aeo',-15),('y','aeo',-20)]:
    for right in rights:kern[(name(left),name(right))]=value
fea='languagesystem DFLT dflt;\nlanguagesystem latn dflt;\nfeature kern {\n'+''.join(f'pos {a} {b} {v};\n' for (a,b),v in kern.items())+'} kern;\n'
fea+='@AboveMarks = ['+' '.join(name(m) for m in marks if m!='\u0327')+'];\n'
fea+=f"feature ccmp {{ sub i' @AboveMarks by {name('ı')}; }} ccmp;\n"
for mark in marks:
    cls='BottomMarks' if mark=='\u0327' else 'TopMarks'
    fea+=f'markClass {name(mark)} <anchor 250 0> @{cls};\n'
fea+='feature mark {\n'
for base in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzı':
    x=METRICS[name(base)][0]//2;y=190 if base.isupper() else 0
    fea+=f'pos base {name(base)} <anchor {x} {y}> mark @TopMarks <anchor {x} 0> mark @BottomMarks;\n'
fea+='} mark;\n'
addOpenTypeFeaturesFromString(fb.font,fea)
kt=newTable('kern');kt.version=0;sub=KernTable_format_0();sub.version=0;sub.coverage=1;sub.kernTable=kern;kt.kernTables=[sub];fb.font['kern']=kt
gasp=newTable('gasp');gasp.gaspRange={65535:15};fb.font['gasp']=gasp
path=OUT/'Thrishloka-Regular.ttf';fb.save(path)
webfont=TTFont(path);webfont.flavor='woff2';webfont.save(OUT/'Thrishloka-Regular.woff2')

# Round-trip validation, complete ASCII mapping, and table checks.
font=TTFont(path,checkChecksums=2); cmap=font.getBestCmap()
assert all(c in cmap for c in range(32,127))
assert all(c in cmap for c in range(160,256))
assert font['OS/2'].usWeightClass==400 and font['OS/2'].fsType==0
for tag in font.keys():font[tag]
font.save(WORK/'roundtrip.ttf')
report={'release_version':'1.0.1','license':'OFL-1.1','copyright':'Copyright (c) 2026, Sneha.','family':'Thrishloka','style':'Regular','format':'TrueType','unicode_characters':len(cmap),'glyphs':len(font.getGlyphOrder()),'ASCII_complete':True,'Latin1_complete':True,'missing_ASCII':[],'units_per_em':1000,'kerning_pairs':len(kern),'outline_sources':SOURCES}
(OUT/'font-details.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(OUT/'character-set.txt').write_text('Thrishloka Regular — supported visible characters\n\n'+''.join(chr(c) for c in sorted(cmap) if c<0x2000)+'\n\nPlus Unicode spacing characters and nonspacing accent marks.\n')
print(json.dumps({k:v for k,v in report.items() if k!='outline_sources'},indent=2))

# Render the actual font through FreeType, not an AI preview.
navy='#0b214e';black='#293445'
img=Image.new('RGB',(1800,1680),'white');dr=ImageDraw.Draw(img)
def label(text,x,y,size=26):
    for ui in ['Arial.ttf','DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']:
        try:
            font=ImageFont.truetype(ui,size);break
        except OSError:pass
    else:
        font=ImageFont.load_default()
        text=text.replace('—','-').replace('·','/')
    dr.text((x,y),text,font=font,fill=black)
def sample(text,x,y,size=72):dr.text((x,y),text,font=ImageFont.truetype(str(path),size),fill=navy)
label('THRISHLOKA / REGULAR — RENDERED FROM THE TTF',70,45,28)
sample('Current offers and more',70,100,115)
sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ',70,275,72)
sample('abcdefghijklmnopqrstuvwxyz',70,390,88)
sample('0123456789',70,520,90)
sample('! " # $ % & \' ( ) * + , - . /',70,665,68)
sample(': ; < = > ? @ [ \\ ] ^ _ ` { | } ~',70,765,68)
sample('€ £ ¥ ₹ ¢ © ® ™ ° ± × ÷ – — …',70,865,65)
sample('ÀÁÂÃÄÅ Ç ÈÉÊË ÌÍÎÏ Ñ ÒÓÔÕÖ Ø ÙÚÛÜ Ý',70,970,56)
sample('àáâãäå ç èéêë ìíîï ñ òóôõö ø ùúûü ýÿ ß',70,1060,56)
label('Reading sample',70,1190,27)
sample('The quick brown fox jumps over the lazy dog.',70,1235,46)
sample('Pack my box with five dozen liquor jugs. 0123456789',70,1300,40)
sample('A personal typeface for clear, thoughtful communication.',70,1360,36)
sample('AVATAR  WATER  Typography  office  minimum  rhythm',70,1430,42)
label('26 uppercase letters · 26 lowercase letters · numbers · punctuation · Latin accents',70,1570,25)
img.save(OUT/'Thrishloka-Regular-specimen.png')
