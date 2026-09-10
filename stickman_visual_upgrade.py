"""Humanized visual renderer for Tennis Stickman.

This module replaces only the rendering stage. Pose extraction, impact detection,
audio, caching and racket-direction tracking stay in the validated base module.
"""

from __future__ import annotations
import math
import numpy as np
import cv2

VERSION = "humanized-1"


def _v(p):
    return np.asarray(p, dtype=np.float64)


def _pt(p):
    a = np.rint(np.asarray(p, dtype=np.float64)).astype(int)
    return int(a[0]), int(a[1])


def _norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _unit(v, fallback=(1.0, 0.0)):
    v = _v(v)
    n = _norm(v)
    if not math.isfinite(n) or n < 1e-6:
        return _v(fallback)
    return v / n


def _lerp(a, b, t):
    return _v(a) * (1.0 - t) + _v(b) * t


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _poly(canvas, pts, fill, outline=None, thickness=2):
    pts = np.asarray([_pt(p) for p in pts], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], fill, cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(canvas, [pts], True, outline, int(thickness), cv2.LINE_AA)


def _humanized_pose(base, lm, w, h):
    """Return render-space points with stable, human-like torso/pelvis proportions."""
    idx = {
        "ls": base.L_SHOULDER, "rs": base.R_SHOULDER,
        "le": base.L_ELBOW, "re": base.R_ELBOW,
        "lw": base.L_WRIST, "rw": base.R_WRIST,
        "lh": base.L_HIP, "rh": base.R_HIP,
        "lk": base.L_KNEE, "rk": base.R_KNEE,
        "la": base.L_ANKLE, "ra": base.R_ANKLE,
        "lheel": base.L_HEEL, "rheel": base.R_HEEL,
        "ltoe": base.L_FOOT_INDEX, "rtoe": base.R_FOOT_INDEX,
        "nose": base.NOSE,
    }
    p = {k: _v(base.get_point(lm, j, w, h)) for k, j in idx.items()}
    sh_mid = (p["ls"] + p["rs"]) * 0.5
    hip_mid = (p["lh"] + p["rh"]) * 0.5
    shoulder = p["rs"] - p["ls"]
    sh_w = max(_norm(shoulder), 1.0)
    spine = sh_mid - hip_mid
    up = _unit(spine, (0.0, -1.0))
    right = np.array([-up[1], up[0]], dtype=np.float64)
    if float(np.dot(right, shoulder)) < 0:
        right = -right

    raw_hip_w = _norm(p["rh"] - p["lh"])
    visual_hip_w = _clamp(raw_hip_w, sh_w * 0.62, sh_w * 0.82)
    target_lh = hip_mid - right * visual_hip_w * 0.5
    target_rh = hip_mid + right * visual_hip_w * 0.5
    p["lh"] = _lerp(p["lh"], target_lh, 0.72)
    p["rh"] = _lerp(p["rh"], target_rh, 0.72)

    p["neck"] = sh_mid
    p["hip_mid"] = (p["lh"] + p["rh"]) * 0.5
    p["up"] = up
    p["right"] = right
    p["shoulder_width"] = sh_w
    p["head_r"] = _clamp(sh_w * 0.35, 14.0 * base.LW, 50.0 * base.LW)
    calculated_head = sh_mid + up * (p["head_r"] * 1.58)
    # Nose x is a useful head-center cue during torso rotation; keep the vertical
    # placement geometric so looking up/down does not make the head jump.
    calculated_head[0] = p["nose"][0]
    p["head_center"] = calculated_head
    p["head_bottom"] = p["head_center"] - up * p["head_r"]
    return p


def _draw_background(base, w, h):
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.615)
    top = np.array([242, 242, 242], dtype=np.float32)
    bottom = np.array([221, 221, 221], dtype=np.float32)
    for y in range(horizon):
        t = y / max(horizon - 1, 1)
        canvas[y, :] = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    court_top = np.array([76, 120, 78], dtype=np.float32)
    court_bottom = court_top * 0.78
    for y in range(horizon, h):
        t = (y - horizon) / max(h - horizon - 1, 1)
        canvas[y, :] = (court_top * (1.0 - t) + court_bottom * t).astype(np.uint8)

    c = (228, 228, 228)
    thick = max(1, int(round(base.LW * 1.1)))
    for sx, bx in zip([w*.33, w*.41, w*.59, w*.67], [w*.15, w*.29, w*.71, w*.85]):
        cv2.line(canvas, _pt((sx, horizon)), _pt((bx, h)), c, thick, cv2.LINE_AA)
    cv2.line(canvas, _pt((w*.5, horizon)), _pt((w*.5, h)), c, thick, cv2.LINE_AA)
    for yy in (horizon + (h-horizon)*.47, horizon + (h-horizon)*.84):
        q = (yy-horizon)/max(h-horizon,1)
        left = (1-q)*w*.33 + q*w*.15
        right = (1-q)*w*.67 + q*w*.85
        cv2.line(canvas, _pt((left,yy)), _pt((right,yy)), c, thick, cv2.LINE_AA)
    return canvas


def _draw_shadow(base, canvas, la, ra):
    la, ra = _v(la), _v(ra)
    c = (la + ra) * 0.5
    spread = max(abs(ra[0]-la[0]) * 0.65, 35*base.LW)
    overlay = canvas.copy()
    cv2.ellipse(overlay, _pt((c[0], max(la[1],ra[1])+8*base.LW)),
                (int(spread), max(5,int(spread*.11))), 0, 0, 360,
                (35,55,38), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .28, canvas, .72, 0, canvas)


def _draw_limb(base, canvas, a, b, thickness=None):
    thickness = thickness or max(int(5.6*base.LW), 3)
    cv2.line(canvas, _pt(a), _pt(b), (20,20,20), int(thickness), cv2.LINE_AA)


def _draw_joint(base, canvas, p, r, fill=(64,64,64)):
    r=max(3,int(r))
    cv2.circle(canvas,_pt(p),r,fill,-1,cv2.LINE_AA)
    cv2.circle(canvas,_pt(p),r,(20,20,20),max(2,int(1.7*base.LW)),cv2.LINE_AA)


def _draw_torso(base, canvas, p):
    ls,rs,lh,rh = (p[k] for k in ("ls","rs","lh","rh"))
    up,right = p["up"],p["right"]
    sh_w=p["shoulder_width"]
    sh_in = sh_w * .035
    ls2=ls + right*sh_in
    rs2=rs - right*sh_in
    hip_mid=(lh+rh)*.5
    hip_w=_norm(rh-lh)
    waist_l=hip_mid-right*hip_w*.48
    waist_r=hip_mid+right*hip_w*.48
    pts=[ls2,rs2,waist_r,rh,lh,waist_l]
    _poly(canvas,pts,(250,250,250),(20,20,20),max(3,int(3.1*base.LW)))
    c=_lerp((ls+rs)*.5,hip_mid,.61)
    for off, slant in ((-.13,.13),(.11,.16)):
        a=c+right*(sh_w*off)-up*(sh_w*.04)
        b=a+right*(sh_w*slant)+up*(sh_w*.18)
        cv2.line(canvas,_pt(a),_pt(b),(202,202,202),max(1,int(base.LW)),cv2.LINE_AA)


def _draw_shorts(base, canvas, p):
    lh,rh,lk,rk=(p[k] for k in ("lh","rh","lk","rk"))
    right=p["right"]
    lbot=_lerp(lh,lk,.43)
    rbot=_lerp(rh,rk,.43)
    mid=(lh+rh)*.5
    crotch_top=_lerp(mid,(lbot+rbot)*.5,.57)
    crotch=_lerp(mid,(lbot+rbot)*.5,.96)
    pts=[lh-right*(5*base.LW), rh+right*(5*base.LW),
         rbot+right*(9*base.LW), crotch+right*(8*base.LW),
         crotch_top, crotch-right*(8*base.LW), lbot-right*(9*base.LW)]
    _poly(canvas,pts,(40,35,180),(20,20,20),max(3,int(3*base.LW)))
    for a,b in ((lh,lbot),(rh,rbot)):
        cv2.line(canvas,_pt(_lerp(a,b,.08)),_pt(_lerp(a,b,.88)),(240,240,240),
                 max(1,int(1.4*base.LW)),cv2.LINE_AA)
    cv2.line(canvas,_pt(crotch_top),_pt(crotch+right*(12*base.LW)),
             (25,20,130),max(1,int(base.LW)),cv2.LINE_AA)


def _shoe_basis(foot, knee_to_ankle, size, prev=None):
    length=_norm(foot)
    if length>max(2.0,size*.07):
        angle=math.atan2(foot[1],foot[0])
    elif prev is not None:
        angle=prev["angle"]
    else:
        angle=math.atan2(knee_to_ankle[1],knee_to_ankle[0]) if _norm(knee_to_ankle)>1e-6 else math.pi/2
    if prev is not None:
        delta=math.atan2(math.sin(angle-prev["angle"]),math.cos(angle-prev["angle"]))
        angle=prev["angle"]+.38*delta
    d=np.array([math.cos(angle),math.sin(angle)],float)
    side=np.array([-d[1],d[0]],float)
    if side[1]<0: side=-side
    side_weight=_clamp((abs(d[0])-.12)/.30,0,1)
    side_weight=side_weight*side_weight*(3-2*side_weight)
    length_weight=_clamp((length/max(size,1e-6)-.18)/.42,0,1)
    blend=side_weight*length_weight
    if prev is not None:
        blend=prev["blend"]+.28*(blend-prev["blend"])
    return d,side,min(blend,side_weight,length_weight),{"angle":angle,"blend":blend}


def _draw_front_shoe(base, canvas, ankle, d, size, fill):
    n=np.array([-d[1],d[0]])*.72
    profile=[(-.24,0),(-.37,.24),(-.43,.67),(-.22,.78),(0,.82),(.22,.78),(.43,.67),(.37,.24),(.24,0)]
    _poly(canvas,[ankle+n*x*size+d*y*size for x,y in profile],fill,(20,20,20),max(2,int(2.1*base.LW)))
    tong=[ankle+n*(-.16)*size+d*.03*size,ankle+n*.16*size+d*.03*size,
          ankle+n*.12*size+d*.48*size,ankle-n*.12*size+d*.48*size]
    _poly(canvas,tong,(125,125,125),None,1)
    cv2.line(canvas,_pt(ankle+n*(-.39)*size+d*.70*size),_pt(ankle+n*.39*size+d*.70*size),
             (225,225,225),max(2,int(1.6*base.LW)),cv2.LINE_AA)
    for f in (.18,.29,.40,.51):
        c=ankle+d*f*size
        cv2.line(canvas,_pt(c-n*.15*size),_pt(c+n*.15*size),(250,250,250),
                 max(1,int(1.4*base.LW)),cv2.LINE_AA)
    c=ankle+d*.57*size
    cv2.line(canvas,_pt(c-n*.29*size),_pt(c+n*.29*size),(95,95,95),max(1,int(base.LW)),cv2.LINE_AA)


def _draw_side_shoe(base, canvas, ankle, d, side, size, fill):
    length=size*1.20
    height=size*.62
    P=lambda x,y: ankle+d*(x*length)+side*(y*height)
    profile=[(-.13,.02),(-.22,.20),(-.24,.50),(-.18,.77),(-.05,.93),
             (.18,1.00),(.48,1.00),(.73,.96),(.91,.87),(1.02,.71),
             (1.03,.54),(.97,.36),(.83,.24),(.60,.16),(.38,.11),(.17,.06)]
    _poly(canvas,[P(x,y) for x,y in profile],fill,(20,20,20),max(2,int(2.2*base.LW)))
    _poly(canvas,[P(x,y) for x,y in [(-.08,.17),(.10,.23),(.23,.58),(.11,.77),(-.12,.55)]],
          (105,105,105),(65,65,65),max(1,int(base.LW)))
    _poly(canvas,[P(x,y) for x,y in [(.22,.22),(.51,.17),(.75,.28),(.68,.63),(.31,.68),(.16,.48)]],
          (177,177,177),(92,92,92),max(1,int(base.LW)))
    _poly(canvas,[P(x,y) for x,y in [(.73,.28),(.92,.37),(1.01,.54),(.94,.68),(.77,.59)]],
          (132,132,132),(86,86,86),max(1,int(base.LW)))
    cv2.line(canvas,_pt(P(.29,.20)),_pt(P(.62,.27)),(50,50,50),max(2,int(1.7*base.LW)),cv2.LINE_AA)
    for x in (.33,.41,.49,.57,.65):
        c=P(x,.25+(x-.33)*.48)
        cv2.line(canvas,_pt(c-side*.055*height),_pt(c+side*.045*height),(248,248,248),
                 max(1,int(1.3*base.LW)),cv2.LINE_AA)
    for rr in range(2):
        for cc in range(5):
            cv2.circle(canvas,_pt(P(.36+cc*.065,.44+rr*.08)),max(1,int(.55*base.LW)),
                       (100,100,100),-1,cv2.LINE_AA)
    sole=[(-.05,.76),(.15,.91),(.48,.93),(.76,.89),(1.00,.75),
          (.98,.84),(.76,.99),(.46,1.02),(.13,.99),(-.10,.85)]
    _poly(canvas,[P(x,y) for x,y in sole],(235,235,235),(95,95,95),max(1,int(base.LW)))
    cv2.line(canvas,_pt(P(.04,.98)),_pt(P(.96,.92)),(28,28,28),max(2,int(1.6*base.LW)),cv2.LINE_AA)
    _poly(canvas,[P(x,y) for x,y in [(.02,.73),(.19,.77),(.24,.87),(.07,.87)]],
          (218,218,218),(165,165,165),max(1,int(base.LW)))


def _draw_shoe(base, canvas, ankle, heel, toe, knee, size, side_key, state):
    ankle,heel,toe,knee=map(_v,(ankle,heel,toe,knee))
    fill=(165,165,165) if side_key=="left" else (145,145,145)
    d,side,blend,next_state=_shoe_basis(toe-heel,ankle-knee,size,state.get(side_key))
    state[side_key]=next_state
    if blend<.10:
        _draw_front_shoe(base,canvas,ankle,d,size,fill)
    elif blend>.90:
        _draw_side_shoe(base,canvas,ankle,d,side,size,fill)
    else:
        a=canvas.copy(); b=canvas.copy()
        _draw_front_shoe(base,a,ankle,d,size,fill)
        _draw_side_shoe(base,b,ankle,d,side,size,fill)
        cv2.addWeighted(a,1-blend,b,blend,0,dst=canvas)


def _draw_racket(base, canvas, wrist, direction, head_r, face_ratio=1.0):
    wrist=_v(wrist)
    d=_unit(direction,(1,0))
    perp=np.array([-d[1],d[0]],float)
    grip=head_r*.92
    major=head_r*1.22
    minor=head_r*.80*_clamp(face_ratio,.24,1.0)
    grip_end=wrist+d*grip
    center=grip_end+d*major*.94
    cv2.line(canvas,_pt(wrist),_pt(grip_end),(40,40,40),max(3,int(4.2*base.LW)),cv2.LINE_AA)
    angle=math.degrees(math.atan2(d[1],d[0]))
    cv2.ellipse(canvas,_pt(center),(max(2,int(major)),max(2,int(minor))),angle,0,360,
                (30,30,220),max(2,int(2.7*base.LW)),cv2.LINE_AA)
    st=max(1,int(.75*base.LW))
    for f in (-.32,0,.32):
        c=center+perp*minor*f
        cv2.line(canvas,_pt(c-d*major*.62),_pt(c+d*major*.62),(210,210,215),st,cv2.LINE_AA)
    for f in (-.34,0,.34):
        c=center+d*major*f
        cv2.line(canvas,_pt(c-perp*minor*.65),_pt(c+perp*minor*.65),(210,210,215),st,cv2.LINE_AA)
    return center


def _draw_head(base, canvas, p):
    r=float(p["head_r"])
    c=p["head_center"]
    cv2.circle(canvas,_pt(c),int(r),(250,250,250),-1,cv2.LINE_AA)
    cv2.circle(canvas,_pt(c),int(r),(20,20,20),max(3,int(3.3*base.LW)),cv2.LINE_AA)
    _draw_limb(base,canvas,p["neck"],p["head_bottom"],max(3,int(4.2*base.LW)))


def _draw_angles(base, canvas, p, is_right_handed):
    from PIL import Image, ImageDraw
    specs=[
        ((p["re"],p["rs"],p["rw"]) if is_right_handed else (p["le"],p["ls"],p["lw"]),(0,165,255)),
        ((p["lk"],p["lh"],p["la"]),(50,220,50)),
        ((p["rk"],p["rh"],p["ra"]),(50,220,50)),
    ]
    tasks=[]
    for (joint,p1,p2),color in specs:
        j,p1,p2=map(_v,(joint,p1,p2))
        v1=p1-j; v2=p2-j
        denom=max(_norm(v1)*_norm(v2),1e-9)
        val=math.degrees(math.acos(_clamp(float(np.dot(v1,v2))/denom,-1,1)))
        a1=math.degrees(math.atan2(v1[1],v1[0])); a2=math.degrees(math.atan2(v2[1],v2[0]))
        diff=(a2-a1)%360
        start,end=(a2,a1+360) if diff>180 else (a1,a2)
        radius=max(8,int(11*base.LW))
        overlay=canvas.copy()
        cv2.ellipse(overlay,_pt(j),(radius,radius),0,start,end,color,-1,cv2.LINE_AA)
        cv2.addWeighted(overlay,.20,canvas,.80,0,canvas)
        cv2.ellipse(canvas,_pt(j),(radius,radius),0,start,end,color,max(1,int(base.LW)),cv2.LINE_AA)
        mid=math.radians((start+end)*.5)
        pos=j+np.array([math.cos(mid),math.sin(mid)])*(radius+8*base.LW)
        tasks.append((f"{int(round(val))}°",_pt(pos),(color[2],color[1],color[0])))
    image=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(image)
    font=base.load_font(max(9,int(8.5*base.LW)))
    for text,(x,y),rgb in tasks:
        for ox,oy in ((-1,0),(1,0),(0,-1),(0,1)):
            draw.text((x+ox,y+oy),text,font=font,fill=(20,20,20))
        draw.text((x,y),text,font=font,fill=rgb)
    np.copyto(canvas,cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))


def _draw_stickman(base, canvas, lm, w, h, is_right_handed, racket_direction,
                   shoe_state, racket_trail=None, hand_trail=None, two_handed=False,
                   face_ratio=1.0):
    p=_humanized_pose(base,lm,w,h)
    hr=p["head_r"]
    limb=max(3,int(hr*.22))
    joint=max(4,int(hr*.28))
    hand=max(4,int(hr*.27))

    _draw_shadow(base,canvas,p["la"],p["ra"])
    _draw_limb(base,canvas,p["lh"],p["lk"],limb)
    _draw_limb(base,canvas,p["lk"],p["la"],limb)
    _draw_limb(base,canvas,p["rh"],p["rk"],limb)
    _draw_limb(base,canvas,p["rk"],p["ra"],limb)
    _draw_joint(base,canvas,p["lk"],joint)
    _draw_joint(base,canvas,p["rk"],joint)

    _draw_shorts(base,canvas,p)
    _draw_torso(base,canvas,p)
    _draw_head(base,canvas,p)
    _draw_joint(base,canvas,p["ls"],joint,(235,235,235))
    _draw_joint(base,canvas,p["rs"],joint,(65,65,65))

    direction=_unit(racket_direction if racket_direction is not None else
                    (p["rw"]-p["re"] if is_right_handed else p["lw"]-p["le"]),(1,0))
    grip=hr*.92
    lw_render=p["lw"].copy(); rw_render=p["rw"].copy()
    if two_handed:
        dominant=p["rw"] if is_right_handed else p["lw"]
        other=dominant+direction*grip*.58
        if is_right_handed: lw_render=other
        else: rw_render=other

    _draw_limb(base,canvas,p["ls"],p["le"],limb)
    _draw_limb(base,canvas,p["le"],lw_render,limb)
    _draw_limb(base,canvas,p["rs"],p["re"],limb)
    _draw_limb(base,canvas,p["re"],rw_render,limb)
    _draw_joint(base,canvas,p["le"],joint)
    _draw_joint(base,canvas,p["re"],joint)
    for wrist in (lw_render,rw_render):
        cv2.circle(canvas,_pt(wrist),hand,(55,55,55),-1,cv2.LINE_AA)
        cv2.circle(canvas,_pt(wrist),hand,(20,20,20),max(2,int(1.6*base.LW)),cv2.LINE_AA)

    wrist=rw_render if is_right_handed else lw_render
    center=_draw_racket(base,canvas,wrist,direction,hr,face_ratio)
    if racket_trail is not None:
        racket_trail.append(_pt(center)); del racket_trail[:-20]
        if len(racket_trail)>1: base.draw_glowing_trail(canvas,racket_trail,(0,220,255))
    if hand_trail is not None:
        hand_trail.append(_pt(wrist)); del hand_trail[:-20]
        if len(hand_trail)>1: base.draw_glowing_trail(canvas,hand_trail,(255,150,50))

    shoe_size=hr*1.55
    _draw_shoe(base,canvas,p["la"],p["lheel"],p["ltoe"],p["lk"],shoe_size,"left",shoe_state)
    _draw_shoe(base,canvas,p["ra"],p["rheel"],p["rtoe"],p["rk"],shoe_size,"right",shoe_state)
    _draw_angles(base,canvas,p,is_right_handed)
    return p


def install(base):
    """Install the humanized renderer into the existing validated pipeline."""
    def render_frames(actual_input, output_path, frames, fps, orig_w, orig_h, *,
                      height, ssaa, speed, is_right_handed, is_serve, label, desc,
                      strobe, strobe_frames, strobe_step, lag_scale, no_trail, two_handed,
                      compare, impact_points):
        base.SSAA=ssaa
        out_h=height
        out_w=max(2,round(orig_w*height/orig_h/2)*2)
        render_w,render_h=out_w*ssaa,out_h*ssaa
        if render_w*render_h>40_000_000:
            raise ValueError("렌더 크기가 너무 큽니다. --height 또는 --ssaa를 줄이세요.")
        base.configure_thickness(render_h)
        background=_draw_background(base,render_w,render_h)
        tracker=base.RacketDirectionTracker(fps)
        sources={}; shoe_state={}
        racket_trail,hand_trail=(None,None) if no_trail else ([],[])
        cap=cv2.VideoCapture(str(actual_input)) if compare else None
        written=0
        width=out_w*(2 if compare else 1)
        face=.88
        try:
            with base.FFmpegVideoWriter("ffmpeg",output_path,width,out_h,fps) as writer:
                for i,lm in enumerate(frames):
                    direction,source=tracker.update(lm,orig_w,orig_h,is_right_handed)
                    sources[source]=sources.get(source,0)+1
                    canvas=background.copy()
                    nearest=min(impact_points,key=lambda x:abs(i-x)) if impact_points else None
                    k=i-nearest if nearest is not None else 9999
                    target=.38 if nearest is not None and -2<=k<=7 else (.70 if nearest is not None and -6<=k<=18 else .92)
                    face += .18*(target-face)
                    if lm is not None:
                        _draw_stickman(base,canvas,lm,render_w,render_h,is_right_handed,
                                       direction,shoe_state,racket_trail,hand_trail,
                                       two_handed,face)
                    else:
                        shoe_state.clear(); tracker.reset()
                        if racket_trail is not None: racket_trail.clear()
                        if hand_trail is not None: hand_trail.clear()
                    final=cv2.resize(canvas,(out_w,out_h),interpolation=cv2.INTER_AREA)
                    if nearest is not None and 0<=k<=max(1,round(fps*.12)):
                        alpha=.22*(1-k/max(1,round(fps*.12)))
                        cv2.addWeighted(np.full_like(final,255),alpha,final,1-alpha,0,final)
                    if label:
                        final=base.draw_label(final,label,desc)
                    if compare:
                        ok,source_frame=cap.read()
                        if not ok: raise RuntimeError(f"비교 영상 읽기 실패: {i}프레임")
                        original=cv2.resize(source_frame,(out_w,out_h),interpolation=cv2.INTER_AREA)
                        final=np.hstack((original,final))
                    target_count=max(1,int(math.floor((i+1)/speed+.5)))
                    while written<target_count:
                        writer.write(final); written+=1
                    if (i+1)%50==0:
                        print(f"[렌더링] {i+1}/{len(frames)} ({(i+1)/len(frames):.0%})",flush=True)
        finally:
            if cap is not None: cap.release()
        return width,out_h,written,sources

    base.render_frames=render_frames
    base.VERSION = getattr(base, "VERSION", "12.4") + "+humanized.1"
    return base
