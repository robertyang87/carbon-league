# fork_d6campocc: base_v53 (public claude-opus-v53) with PARAMS camp_occ 0.0 -> 30.0 (D6 activation). Lineage-regression activation scan variant, not for submission.
"""Carbon League strategy.

Design (derived from the environment rules, not from other agents' code):
  * exact carbon bookkeeping: unit/tree positions are public, so once a cell (or one
    of its mirror cells) has been seen its value can be tracked exactly;
  * collectors plan with a whole-map dynamic programme (stay/move, depletion while
    staying, hold cost proportional to cargo, absorbing home bases);
  * hard one-step safety against lighter enemy collectors, soft penalties beyond;
  * planters plant / seize by marginal tree value; collectors punish intruders;
  * joint action resolution so own units never collide.
"""
import os
import random
from itertools import product
import numpy as np

SIZE = 15
N = SIZE * SIZE
ACT = ["UP", "RIGHT", "DOWN", "LEFT"]
INF = float("inf")

_HT = {}
# Solved three-hunter boxing game (offsets relative to the target, Manhattan radius 5): chance, discounted
# 0.85 per turn, that three lighter collectors box a careful target that picks its best safe cell 70% of the time.
_HUNT_BLOB = (
    "c-ox3>2}*j);0XQyJHcYFi%y5c_tyw6vaW5#aNWqitS|CSE^g_)%3RyKsk0|f9dCSm~%=bSghKpI@1QhfBpOFKfk>BrE_)lrq_GZ>2%&meYt6L"
    "Bt@o{)Kv|xC__B2X`0-Hf*4Bwdh_N#Z{A#WrC+4()m6VQb)<fu8tuQU{$OC}#$eDNXqv8R16flvSytpBKJGuwr(Ioj-t^?Y)O>6YuSxW=Lq*|+"
    "+Lb!Z$Mq$AT)#W)ccm+gQNloby(>wQt}wi&A$4`FX?0aIHZ)lp$hx6xT__UWtIkyiZ|?Ov9SpLor~_*B`#whtgS7lgFm<5mzVBNylqD<r5M#p#"
    "n>XR(@jo?W1(5VRP@Dl0S+Q+q+lp?RwyvAHO7FRnx}DBIRk}cf8r-K3>!?grz6CTae2lK)Q!#2CXwY=7Iv5|O-^{CrF?4BOvS!34fRHrIM;<`Q"
    "gDxI;rqk<pE*Lb81_NsKevCm6@9RswzBEv@AvJnKpQF|95A9?knAVrgFbLeCW|A-l-J#sW3Geq6wQ0yQ;F2*$g@tv@Rm?n$LmM-PFfmF@y00i*"
    "MIoRR$8{aF6weDp;CYJadBXEd)5CuRR8Q*3s?x^?P=ovEzb~mgu0sY`o33re4o&J?SEc??QwCDMM-A=|F?>m5Vv^XHfg1#&DMOjs(9niBr57O4"
    "_zuY4-+=_|5}i=`*CN>YIf7UrLxQ*~bEXrBatmFzg|0`?^)v&>nYu1Zy}>~78Crv(!n2#0>qEs(lEfeCHVJD8ur>LrJJdDDL2Ia+JbK9^Y#J*X"
    "k0af1#<8wD>JUT5aBSDL(UOC}^SvOD&A|8kz%+gQXPAaM>|V*5={Oi>_sWR!mx!DCkb1HVJsvxT?&yTPtn18I27{iY^Y|v?%5dOgT(&aw6I&lz"
    "`f%9k4|T?6T4*iKSOYE2UBbC%XuWO=FCq5|Z|AS@UIESq)i<4<tVd{G(VH}l-b(_n%giBIyRz-&%+(~>V2rsiIgRM{6yXw@UBVD>c3TX4gkev&"
    "dS=F4N9*<#k-ks3gL4v>W!zKkM)a~w;9Qv&PhXP6RZJ_szK)=-BU!TsrXovjlDwqcGKbOib!?eRbPZu?GT`YrJb9lArQfAKfk9l_Y7Y+>1wcs9"
    "b)^=%E<x9`bnIo#Ww~d|-HwZi82A7NV}?f;Gmmt+*Lg$cQ0`xmaqBcoPO<C+=S@;@ie0Cx*(9v4Dw0Laovz}<NfPulrx9I&#S}ZL>LgVasJc@H"
    "itY^4_cx2g5+a!`GD|o~W}3E|W{C~FA`H=v@SlmDDuk6SW(ldp5f+|PRVApZ1YHJ|qpOY%TOTAz>?qwJakL&DU2JIARwT!@wXUUgyEN3`&!IYT"
    "l>`cnP}ZD5Tve`Z+m&k?wi?H-8#<wDnU?C<ZiN?Y!^B6TYsCT3U;BLR%#K`DwTHHC8J4Qqsyd`sRZCKb(!a0hUq>FiQHC8XxWPa+6>4ajWum3("
    "y3GR{M#CU54N{h>TQJQX_yJYxXe!R0t1I}Q8m@z>=^dM~&<`1`Ip+cAq&OLO>xvdJIi0@L7-v;(8dA^Hn-<JPX<>m56y4JGE|gB@g5D$#IzX}+"
    "q-rWsGswV03xl+xYmd|D;0+ilMuWr{n-NN{Bb2_5P-;dP{a}P0N!7Y|kEAn4F>oXraq4ok+!z)$6~bY06nnyD5riuo8$*zqx8S7e&0Ac~nvZcf"
    "FQW)uyI~mOoo*xWG30ax-4+9-X#fMGY5j?TVHpNtU=Rk1tgEUtRQif$H4Rw_JWEv+MYXe~FHA)-;~?M)EGLH%krl(xC0HpeYC}WAbwbZE6wEw`"
    "uTzhv9H+5wa~As+!#)Mhs)I>8Dm4NDTcBbs#&M8@q2tD3=!MJ;xyKxCTN(@(b5ua2pwU7!aO1p=T8Q%RAW{c9bDM()uXy%Z9KRjW!9h`!zV8MF"
    "gy&(|z5?0(7Rc!jfOM?R1wk^MErOk&AvhRl28=d29o^@Su1P(`A?R#49fFRW4usLuEDxWH_bXNyaAqs2CaSwMVAzy-nv#hiVJk+!UHfXF*#--^"
    "V`;YS#V<w1EZc#rNT3>FsyI>DjKbsCKSL``y|9iV%SrtpjD+K3I@nnzPLyhRk!J$=p*J4C%wwE{Vza^WW|NI4m<KH3xEvlO4s#8e#l^!Y0v2&m"
    "RDdsj#gG7xVjQEjIEO#P8LwM7IURx*&$zZrI|z_B!gVa#uO@k*#_7cHNaS?j%D*}ttacw>N9(|{hEt#0l58@cbHtB~H%`Y22t%n$7={7ssttxc"
    "!Vsco*P|$K;dK1`Ls9q?2*7^AHaDVI*HM7z#pd}-S$c|1PA66@>pGToOLiSY_C}+ZW#qb6etSECCC+al1h*{LQ2e6$W0_zw#d<yQ1r!8<_zE57"
    "Owb7i1R-#w)yEo}B*HRva!0OZ_=#&9c#C1VZUQT)4zZaqWw{aXJhB99I|Zte>M-uR0G5;JYJPrKmO)o_(;PljcQX<_zq%h+fu_0ncob{8JBqQ~"
    "hxs25yWIA@WCkTzl3VD1xm<c~o~Jof$`j9aEF%ip^zMF2YU&5HCSdim>2!h<kxd~ySQw*uf|TzV*!V^e1U@qn@Gx7(5|U}CSwKT&G*maOr?E;S"
    "lra|k5jc)B3M|Xfb99{W#CIGU8;94P;6U3nscW<CnmafZvfOrNcX4V}&5;G{gbmAMTROd}+Fc7FdgF73%9{Zq7R8hY11ugiOp9E3M>p+&BLRT1"
    "z_~G0s}Zn$<OC#nD8UOzCc+KJQmA2=5kpHiEhpYY+|W(tMZO0`!a>&vS;=>F5lRhY!&0cBX>ov-1_zfcBHu))7es*|1!2>N$;uTX8cD+lWmAL0"
    "!3Z&Ftz%diVE}|p7k11wE!S}^3x9XIL!~pI@X9h|%!(R8OtX?qKUyHD(KUgOL3}Dm*rrB?TpHqI6w5IOmPrj3nFd-0k2bM`Rp|MV$4EgqCRQ9U"
    "Ks4E6pmtv|P`f|HKpv{HJZ#^9?2g`$#c0tR(oGUtI5LEhCl%5gh8k?DX$^6_q2PGaEK|(Muq;s=YeaS~%W8B16<fAz3~^3W0|yu?HH7aAw2Uyk"
    "#($x~!vKa3o|dd6%%MDlWp>P=Wzq>rOCW3ZthhcF!m^AQT{G8=A#n2Xv4}1kaE55a8G+NY!eo;YI3dQ(NL|1|WYYqP#y<gx^#BcTr|GJO4uGI;"
    "7(RB8flPsn7!(OxGgSkdQSIYg8d*$E6)>gSB*Hm$!z2oI!zJM$a)N&}`@q;LGlo3Dv_yHVQy1Xob(TUkLboi>ak>b}oG9>Mgl#6KFS8U*lUJ+N"
    "$Yo+Ra%_P`2disX0c<9nf-IZo^DHyMEREAF3{y7^9ZGrBf$exf<k*_p73u6{DwwT~R;!9LceH{x7eL;`5Ls4YaAqGn{#cTkOS3v>q#*FnHG~Xt"
    "W?-&;J4&`Gu+qaQ30+byG9OCiVBSL<i0#0D7-9{bb8-Yw5VX{h4-Y(w*Lwq5V>#mY-Vm@d7x|D2q4nYIFnAq7jM0J#|4Lxu^GVpMcEL~whIe_3"
    "Vg3z<?ySf{rmDKP`BWE)s*CA-KGqD;2z53qQ%?nw>o0Yk7@SY``!UphjPt6&vJaBF{;~!(>Uh7OaBhtE5EL6|YCkFJPa6-GXFQ)zg$^YQ4G&cV"
    "bg=6JA!?fC-$8c)%uo~p3^92aVSY9`K_-?n%7l=~AW7>Xsu}jFXn?r~lO%EaDhtBIhu;n%+=1!c?&=(K@@MmTsTn>APxEe%G#AsKZ8oKD`XoGk"
    "^Xa&oIG#5>?iP*^(}isdeRF#o1h?zy8fwa?0qm6NdCp>YoRXR%BoMZ(n?||W%zRQlb_Zsml=Jz_*VS_F8=7ZmnxDXNt9``ui18=_7eQDC15<<$"
    "Y=E&~a3+n=^fU|%(T2AP8CLIX76cP9fl244FG9pHLfF`9=-Sjg6ES=!%mFh8Upo_1m>{7DEMI1t!qgRdRcC#jsv!~u=#t(<+dA^5Cfau4$|$Jk"
    "Jc>%JKOlfG(Iw^Dup(4`dkyOm6gaGn(cLsg3rD44O^KE85ZGg%coB1iYj>`sHp7u#GaTvL3@7?=h9fmT0BNYHHl7U)FK9fQ!M*DUYer=Vvf$L1"
    "fAMTmzbRTteOYa5aD7<jUQ=t)t+J`LU<V*{hEm#M0A{1Z^j|ROe#M~MU@-j)1{CILvY{dpM)>05B=%pYKcMN4-o}`>@0UdpKtW)AD2Co^j(4ze"
    "2bbfG9vvUmc?(zm4P1)cL1+hb9hDl`bpcuq40M1T!ePiB!q{aV$2sBTyeS&TPS+*u01pWUTDF&7#{ppVk(vNeKV}Fakeo=WYH7%Q;QCa{3y}N_"
    "B~$pz%!cvcK@gjyOieKmjUnQ1fkeaj3y=unkR(XPS0HUou^k;54<aih6hP3$>>C6bohX+Sb`E^e1v(9f#Ze1g_(SL%3vX9s0|zD%LL(d)jUI~T"
    "z^W)>IxwaK)xa~<p-THQLJOe9^Oy11V^}JiO~J5>CXVoI!_s;~ghJdC$lk3eszWbLEfK-7rVh@qkHI6H7+hHnysTlP#j@fhMw+K3<1xH~m5d?q"
    "FbBwfc@SC(a9}}tR|$A~xR)H}Do7oyC|yzD6CLhT#jj%%&W_TP8l2@WzZ9){Exb+9>H=@)ukiL|MYgFVF@yy~@a$4e7+c6`6vfU4g^vJ}{A4o0"
    "iba%7MJdB}NjQe<M1&zuON*hy1Yrn8f>~}o!u!!mmVZ=*?+AagUKda#JiD#~A0e6>Enh|>&k|yCdz(W+aDk{zm%Ork$+3GC$}B8WmZcznlrbN_"
    "C_d!A((h7EV#~dxYD+kH3D%mR>q-rDYL}qvIaIW=xjzjZrQ0EoXvz|&(4r|?=|<cZt#sqNqLpDYFAoBjQYXaevdu`ivKuaURp}d+KbcN5!|_RY"
    "Mznkwmp&pCdKPM$;r-U~kNdkR3!^Eiz@JVreJ8JS*IOcgfJ%5u>7D`WlkaAU2bC~H1&&4W@MxU7q+DHdsc03KaJXtW&Ro5RN6+<<6RVO7@7Tqo"
    "YjQt`1EiuDs^iMdSj^WE=5@`v@M;d^%`{~`@&|@bLLLx|5Uw~D3?Fr^y1G1MuYJ<urOrgk5msV_Q2wAhnkw6P)v^)CcBw+&qyc{kff0!|A+Vin"
    "l7K?sAPIenXnl%k@urK-v;?9FvYE&c8PL(YF1BZSm!=N3WRj*Ksn9|4LhtF*drV}EKE1~jaaz!Oc!JR3U7oojNycTFh$Jo(#JNZT^`3qy2HCI0"
    "pvFDH3Y2+C#gQ;BVmObGM8F&>{`|ibgJfl(*;EXYRav9yD=NatY?*{6%w>$}BY+|S<4DDzX6953YG>}io?+(Maoo;)oJLS-L^D^g_z?%$zN1nj"
    "5P5=@3qxLhs49oYab8s^TuqT;sbSsptQH~#jW{J(hde3k`l%iRqH<i10Z}n(AW{+AxXr<6w0QOb<J;4_I&;I#=D1iS*=7b&#zFTL$nLj5-n2kw"
    "837q&Bmm_9p%^qAm<TVLVvv~x%xsH6$c3D)+(6gUtiY#Ykj^4lYH6q`R+f0ehA$TtFypXbIWU82gnB6kAv}&@uq>?JAbz<XGuvg+^*DoqaF~T0"
    "N9mfd?Rf%Y>7=Pwp0BShmPSQ&owFzcdVyzqo~1fb?Byxu9M$#9`Qt2Q<=eL(VZd(QLQEG}f40oLI4LsMRDkoti2#rPf*}DO#d+dtg9hjA4{=76"
    "fK**=YHfJ|C)~grxPTV|KJeROkn)pakQpPzYKlS8Ch{FgMi%CKNXEF=7K2j45GRugY6<VEZjK$-(_IqZ7lSG)22ou;rD9O}Z81pXFI7dwpyc{G"
    "r(#f!v#aD$XcAC|Ly?O7bUFZ|{8U4X<0#~2I3E9T*cS;aj>r9co)*U$#5h6lAmZWyVRdB3v)Qc5!nhG>;17@yFUF%-7(rQ%)6fe_2tVYA2iz=*"
    "Y)f|F7O@2!+6NA{f~4DQ3P&cXE=RCK>$WKd>3;F9Ee7RznNTq(X^TNm@L({yIrfOAyW`~BVo*xOAdjaxf)gDRidj+Q#cj^=2wn%V^P)~}-aYLX"
    "b#}9z)Y)W|#Ic_(7V|2LvIPXqg3%LMi02`YiMt5Nl7Q<l;M{OIavGNlDh9ctEe7TFBB5eXGnNd0xGn}}TdrQrQ=WuLip^+dk*E`dLa}du(=|4a"
    ";Ua7J+F6#Unkz>k8FQZrxNHwOCKZEd#WaCAMX-p7Srf`~cT}*(ET_c*D;ptZUgQxe7$>Zro=Ul6ucvci3l86;i}m5K7J5&tx5vBVb}h2?cC%e)"
    "*?PIo(rFrXRUL03BjVeRFsYFjX^fVe<?nFn;h=DW6IiJ-P6SOv9(KtLUdkrDS0|(Oe(Gczv)tBMkn&yXEQqG)4cRVj>MU{kO`V0z^1!6D)<sNZ"
    "B3>L7u1V`AC{L4i<`cx)P%+J1k^5Z@E?XcOLu0a7OweM})q1&I%~&#BEmpH6nJ%VDGD$*BQ{bhU5T+`d;_!5TC(NPPKYu<yoKB+NKi?fr<9c`4"
    "kL%5-Qf1w54#qV6%?Ws#egEe8`@7?QEf!nsnVW2}UEgHWd9!j`AbbA|WLsy^dQF`ri_vQ8EXyFEIt#)Gs<Vi%br#pqFLf3diwH0QI%Lth8>ipY"
    "S#)eW-)-tF<0|kSiXX?PNkM0$oD>;UBXk}_zE4+O@T!bFfh^3MzATra=LOYjRYR?6Ap(H~ZOewUI4l?3HJ#P&u-o0=yZLIjTHUAla<xj+X+j)?"
    "?S#w2a>;Bx+?`(z+tt$D?#?$jV7q~s0(@)^H)sI26#o_Q#s2vSz!wDg6o6xnz<D(l`k=vi_hX#Xyvb7m@7-vOY$GnqB93`nLJ+)EXMxMtnmP-g"
    "0M*`5NA^W^7WQYI1z=L0CEF&Eqi9b2eVwJn5b?qIZP{X2euE)<#PW(&Rrj}_w#$jCi`jNLr#cG>^~2--Hc^@2+b@rgO9a=;`^Q_T`&-1==6#+8"
    "qj%>o?;r2)qtm-H)b;5t#O?n2It#t>`1ta)ThGGd@o|4MuZ{<Z^}H$p-TLNvJ}ojfTi;BIBAs1B6j_=Ky!m`rr?GFA^SVewj$k$7NyLzGB@3J-"
    "rUVHxNTvk{8XB{#7;gNwD7?PPqPPf%haxdB{nPKq?G$s{9JdRck{c2lJ-eS8CZ63dbklEyzI}Oqw{v}Ob%OH5>egn$czk>;<I{GveXOEozN-8*"
    "^ta3G?eD+scB}E*4a8<O9%tU>cwAxvo84s=e6!szi!j)1izvV&$fi{h2t6wyd~68Si;_(S@12ALQw-V>nlwVw*H*{<$P?av#9d)8mqj$qr|8V{"
    "iZt03zyo1Xh_1sB&jwkUggVTW{@R&Xmxd{{JmD_4!zl25dez{4VAbii<&v(>7_oNLC`K&{nib?;g&V;pd!!%`i`lTS?OR-7>B4F(Y{igcNeFT7"
    "(g<$}Q+R*R4M}if1z`ygUtS{e-q(n{_tS`+su-$b3BwXds`~VXamh@Dd<CssWY|H4l?z3}cgVtCj<>YI<#?+(zC29}%JqwBk&sFp&!fN>fd*3<"
    "C3%{gW?AZLJ?y=8wnnrS=6ROqVVH|N<bpezrddo_2DZ6kv2slF9k|8xR1!o9y2ex!5+2ofRrB>=^Kx*X73JG89h?;<_8=4ll^uGoK=%F_NTf)J"
    "zx7@R`IUvpUZ*Of7c>Y`uAc^^0wl?{+d!xHbvPS>4xb{zA42E4*gKBgU{(pd6DMVSo)&@*oK_VbSTzR*fe=H@Wq>OJD2&JRFN<k}1n_#dtGV!Z"
    "6DFubN~~eekXTxZljm#vmpc*BTX_@#(}>DwFjO3ANs9ioVKiNQna3DMz1v-Tf=_n7;J#pv>tO3u4TiEPrspx?Qej+xuX@E$Cqp(yV5_${hd;y_"
    "NwVd*dWYZ*JjUE0CEC{|-jb@dHdQe)x+X=&85B23LRAdcPyFWU9|+krtda`Bre%5MU7j(^Vp*Q0mc!E2giAyC)L_Wi1w#y;Fl3^r9C&oTo>j=P"
    "7)y4aZYGr@{N>Hf45|_KdOa>Pi}PsrWj-G@7WTd-3tLka1Hq^~n$N%NfS*<0-!FxL`jZ8%CO4mU#DdOlZkE1)g22NzFVJDm1f75?h$)~sWBhle"
    "XIW;NHS<9h?^v#bnI_2%A_`Ro0Isgd&2?FL$fm=ihoWGz#iOE#6-pBohskcVf4;dcEIU};tY?-RtY@a{n|O9}Q#dSG9S$=n6286r^!)sQ#Bz0h"
    "en4uux}_?{ZoiMB{cg2`T8UMJXw-}%{_uQWLEWxiSx{$oIINHe&JGYhW?{l{tODDbtpnHcUDJ$<BI1?|*XbzCQDn=Didn26hk_6?#)1(t8X@yd"
    "3=CV>J0KbI%oRRrAr^2g+zh<pfETb<OzN6hK10_?eH6hI`L#1|8QYGhdmeYV$*})*de!DKtW3A%{l2BET53x&0>qX!vCknTrcf8Us;NcvUTYD("
    "ueFHYPiqlmEoH?Ng89BK_f15}rwv+Sy+6MD{dnVNH~S-8N_Mk`m`@W0#!_vL=cY0=#ohhWL12-bP7mkL&wB(|hr8!}J>DO7^=MO9m<QY`#iJTE"
    "ns`*}ve|08T(71qnIRsX{uz%dLk%%5OpZBFO}7|fW16t7lT#UR?#2jp;Y;FIhaw{~X?Z_Ix4-g!=`OPy-VZ_1H@si(!VyY6?H_^EncX5tm!c4>"
    ")MIfx9%0+MSg}P}BDg|yMF=8-@O?uY=wiA8bcP1#c0Ys;dow@|ZZ(*p^@^Df%#P3d1!1-rZKf5>isK+fTq1@H9ot>5wuhJVE=*UeoAYi9wb>?!"
    "Avtk}E?+DU3&)V%`}-Xt(v_3muMm;uIBc0GNNjX<AiSuoI9=u&E|)J;tT&@xttvzU^(vr?MiJBN4vhA=;r;w#avBrwmlYFo09o<M`)LS~THeoz"
    "fOE_HrSZ4CUr#gRtmXZJVA=A10bNalT<pU84HOTtm!owR#v=q{iK8*jk#A}YPDpd%tHSJd1~5%OPV3!#?(r}!>UqhL)0QksV0!i|V92s!I)A=d"
    "`)N^3U^<|#XK7X>_mA5u48z%a_w@33EUWp={_%K(IzX(;50B$CiSF+o-+wv38(pJues_X;d%E7=UNg%WEFT}gY&p`X+sDRQ-d?Pwsj2<R5<#wy"
    "Om4p1&V;Vc|K$CAaw5e0!HIka9RN$V*@gFuzvKOaV8R>TFQBU@;g~PHANR6yH=TOAfE%4=P-UibU(c40@Algw2#e)@y9uJA5yAHH`FKRwSS*jn"
    "4b*N!Ks>(u{`=eUYPb8XQ5%TOczZg<6(aU6)GA*_Vd@i+v3qw~dA@tQ5*{qPZJ5Slf86+_eC!V5{g8dYKB$WuoE$u~pg?x7_5&h{@EG7UjTV_6"
    "CK#$8>dgodUSfn{5ny0Rnisny&Z2O&i3-G{kr#xX?~V2dRFPrX)U~PMp&@oL2f{?3Nf}aS`8x5Ds?jx4+%|;?%<-zrHACw+fnxWyrrP~lQ|&f2"
    ")tgIAHOr``ngt9y?3M2FZ5N-@Z<AEC+^l|gdP~{qVsYGT!elYqWO11V7rKj*S_}CTLM|k=R?i*RPjwA*nvCl)nbuL!Cbfzz^;8FkbAuX5E&{Zi"
    "s``MuEOyhX%Hd?v0{#i!$!+MaHl9O@`!${$A2W(uqI685T$EnoIikC4Y$TcJF6=j1l_fJwmYJ=oD$H4Ik+L)u>s9%`(Op(UcS)-4IxXGBSdkOm"
    "<pvFcP+5S`4xzV2&<!*<$>XSnZu>*%Jd4BgDJEj71U^tE%uWyG>w)pt14R(oD$ZS+dvWBbES-HkztnY<-`;+F`?iF7dy_AwnZeASf;a=7fSFzE"
    "`ubdY2xar@s)$mSS5buE*0uG9?uuf=L5q#AUt+SX+4wqQuAPk`5CK?2$rfIcS&BB4H~_B%3>^SRlne!dhuYDolw%Uz<%CGNi0*1|mb-pn{m*pQ"
    "KrzjR?vgD#iZTSen(N2390^e}eI~q0fQ4;{cr;$XeR*LDpB+HK35LEKMo|O|?Fc^op@>D8RkPVFCKU%rc=LLccsfgp{o#-CI85jBJ=ul%>{uk@"
    "BCe4Xn^ZSDT!1~CPI-P?pYnK|NAQM{S5+^$ux%y3zD_Js6qUnrHFQ_i(p@QC!YudFKj|)>5L1YigWMx`;cCclHguQcN1KS~F3*pcB}*Q+zM;F6"
    "AJARQPhz6G;5w2x4h$ZUaJ9I;-Hpo753~7VkwGmoa>9$-hq?@-tfFV378wC?_~Yr_G^-c;r*}75J=xwQqe+}s1<P-X98o9|;GBujF<ZJTCAur+"
    "9$gQh>pHe`?rKo4bXQX5G_B+--4!AvA-c<LsVP57lO!-GmM}b5`+kH)U5hAAYT?Js&T^h6VS+f(HC-2<fPWTTLDJG)%8-fIKEs6R=q$=aQx<g_"
    "y35s6*=gu5Z2PVvW6O71st4=zLiK2his<M&sz)WNM>Q1%F8D(A=wGQG{ePi)w69c;)=)k5h3XmogzD-3KT$oyf24Z4-EZytO>5t?jI2|T!8+le"
    "b9L2HJ#;Z)&!qi#)wxhToxV(0kJD8YB%}+~)36r})zgtKRF5=lphN5(P=&DDQaxASP(4OV_2@=J^+2T&>^Ph{`%3k&@2DQ$Qa$`<R8RfVP(4Xa"
    "R8Rafs^{g8hU$4B65!@7#Gh18gybA<Bq`tioa%{)>XHAM>Y=MerupOtRL`5MtCs5d<?2H9yy-Pm4}GDE#8u})^;}68s;4t-s2)jyH;@!V>b6u*"
    "N4iiwQcLx8iRvL*z%Fw{%+g<}9@S|?uPk1vp7YPCp7Z|?)${R(R1ctwi`#ng1FA=UrFyI%Qa$lcsGc$m-A+sOG~9)xJJN;fky@&!OGFAp_a{^j"
    "5qb^PbNi1}PuWsE<+iRfY_TZz<Cf})zfwJ6OZ9}Wd<<);9`-HO6Eca|85-($R8Rj;s>gJdpHV%3cKt2Y({$EQJ%cE;O<Vbf>XD3rOm<0W>=IVZ"
    "*UDXEm#A*vl%FQwl%F(NmJNjHzR=~K!6yp}#$~=ftRFvZ=OW)6w%g@A->jDNbT)}~UA2TSe8iSROB|j~cfP4KT*vRHlV2a7ACIR+eLMg@Bw?xo"
    "w?GFgvgvP5NbO9ysliifM>Tk)c8e`iyD4SGuVLiypTkI5=_@vU6Agx1A+AG)q1FQ8I$)taKj2V;H-@^kOI;jkW=LjX=XK&bLPu3?qKRPPNmvT("
    "5>^~`p3>IoiGpsp4y3Zgb!b$lhfo`?Lqc%DrxkG>!Vjh|t5xXx5pf+*5MC(gBD|r=7KiOpaML|J9X@=yzZb>n>GARLwm9t{Z}ZzFLQ>my()D4x"
    "63j@CpI^SboL2nZ@1I|u-`&3Z{rTPQZns;Y2i9;MWa$a;--zpI03R=o7p{Yny@u;R^#2{#u`LPbayl=QVas)JqYrrRZZ|}7OsDIj2;(V4koYcr"
    "bC9@>L9Zt(e#3PPG~RF>TFZ6FLo-F9uMLN4YvhLwpC8J|I!VZKiy<<zmg`W}hU-uXLkOkiI%HXExei%wxDHtta}x3^*C86N!*00_o45}92V4j6"
    "ygt3Ve{8sp{K9qYU%8HDbv!~4*8%9(hv)lUk@MO5a8niOY+bc1f#%IOyZs7Lc{$(E)-LB+<P*mMi@CTShrSjhRha~OfCxp0@z92bn_jq%_*<^S"
    "Xt|D|+As;S?%$A*FI<Q2-<?iGbDW4@P=3I5aN;_y)6+dt>g#ls-(RO$6cEYr`)_+vy9?Km#~Vxy6WHu8v+!H4g9wQzY`Kn6l?HlNriiMzp{ri}"
    "H?Bh^t^-?D#=yo%Ax~rCI@Zf7p60WPzL8X<$-cm%`3u*Ph6YU2-?)yn;X2~)xDJ(QF|soZQG~&%EQ1!iyIb!*{JLh@-Fo%wy2w_mCC*6>JEJvr"
    "Mp$Ic?w+2A;u9w<`_IoOQJ<dgpHDUH%(1RFb){*Ioe>rpoV(N0L0Ciobo#xirY%lKB=v}3aqyI?X@5&UyFX1oFBN~~YsFuoihp>QrZ$!O({C&O"
    "`d=&loCiNNxo%@;E){?5k=Kg9|JRCtV`qjMI-1w8GY<EH`Ir3`(Nl1}-z_{ZYD8ma6pJknhb3p0bANw0obT`50{U=Y6x-dl$k#d987`vr;jn_I"
    "i5{N6oIgIT*wb&neSA92o_;%@X7|V0j3Z3>%Bdaz`15mXX8`!(*kCxWbh;$^v|ACUx<CGYK0j~8Vox~di~Y?ppU(5-0xw-+FdE;Fuvi(Av8sT%"
    "A<=NEmx}+8D*pbZ;?Lry;?MtH@jupW#lQZh;_tXEhKvO*hKzq-@h3Y&760h)>DMJ)=DGa!W=<9Vxh;H~>`Z2H;U9j$salPlIltPObIYloFPtju"
    "OhtC){Oc+_K0cm)-Qv%iZDVKPmJf&L`C?kI*<o`tEz=nZY@uY}Ax^Akb!3;Dnl6SZ>zuyr>NBURZjw9}N>IfKT?!NX!f+V&m&20>*qOgq{GUEQ"
    "@1~X=9G>?J%MJF6rs9uhk5el3KOYyQ7F6*+yfhX6VD&^5e{Xe*<zx}3nmiqLhsP?~T{u<PkB85{o%gHjx98pJn(WLZ-8?`4I-QI+&k#k8kQ0Vy"
    "J4utxu}qU9Av-fJQs2suS6aSr8I~89n@nL|(pLOwEEb~?GT&TdKT^fN=FGE!TfD+&6Q?>Em-Goa((7-!hPJ}UGG06Lwk1{ki^k3*5jF|EYIB+R"
    "XJlqDGJ+e_pbH4NF<|RoKkU|bcdS_dx?105#d5XE@>vT1ql@NXYRtBa2{(FTzWwxw_=nGz+wEbK&zGxBKEc5~?X?i4eOrjCFNLVlw}mLpwQb*~"
    "i!t0kQ@!Yt>P4fXN+{J!DAi*e!Iz=+wKn)NR0Lnk9|m8X!wT54*;r2>Ya7<{@zbw2etvh@VpTT>=CeuC7>-Nv8`*)oT&;Is&buUCtybs#9%{F5"
    "ir?7mRQx_1=wb{PhGTc2i!ttZtKEH`<LI2O>fxMfJC}<wlHK{sZoOK%`~4XQq;N>>#Tdxdgx>&8LEf*x@a~S!Z+_h#?p`s>6Ake|gR|Cea9;ix"
    "XM)$&It{!L;7zJx6vaGwg|{y|b{gAiPv+do-j}7VA(c(a7~u(zHOJ9R5p1Ra(+tvJ_GvyBEY6C0J|;D0VFL59BMjZ5%+{Z7fMHq9Zf<5!7-wE3"
    "3(RC1yAxpe<+Pt9v*q%%-$Ox6C-K8!w@KZ_a&h?Y)6+6pE*Fna2dKwGJekCcG_aRXPyapO0K|Qdi{&yoJVM~LI2F!hfBJ&{CjjxZT+WjH9%3?0"
    "VuJ4G(>f~)mqt^SIGqqxo+kio{%MBknB#d}Ae0@GNa*VkW%)jHIG5uxK|-&^$QcL&4UVGzW4-otFBnzp!r(zs7|hp;^6};Iu#V$u`*=J;L4;*k"
    "<JreG5ZFFG9-)p$0^;xmiE>n3uRb>F<|9OOczmqu$HVIISQo2wRRnS1)n)L<%g5Dq6@9#cz#}R=SjH_Ta6DdS;UOf$hL7<?9G77n7VByhU^5rk"
    "PxO_rI2nzPg%Kp<!tg@<`w_10AIC{v+>I#`S#3x8B3tCh!bFvf?}|L3r@!fXoCHSb89H^QqqBT7_F+|XWNHBmngGSqbvMvlBuyG!HK4=msoz{R"
    "FgZ1hcUDZkW3x5=waTpjW0m>BgDG9&!J0Y_i}Ug5emufbEh?nKq*@-#qhv`R=wh{_WJwp>=kpm+qn}ftzhC6X8weyzo`?eGyz*eoMribmXy@P~"
    "G=dkV6zsSdBLx!y#$+<SIvmy!B97J5$McuxhiG<s`uN*#PqWh*;&>0YZ^5;4iXrI2jr)5f{rC6IUt<UWCA_Gfb52(cZ1-R0fybT2ZhsByz1Sf%"
    "!9%!C!-F+f4fy}lRRh48$jQIrtbDy{z)sV<u)S&^{qCv(;=!`S(T9qN^W+8w)(~)?Re-cC4h-fXy(1pX)a?jv7ipwNMc;Z;MoC_mJ`7h*JQ%{B"
    "V!D_=9}a$6PH#4c8L3&Cw>(&$gsWfA#Di7U_SX|ONVUCreC5H)?81YcTORC8JlMJ6!RV?1;=vBS;MQMwFv~HN$@14v2fCsPrgDiOZ{A)tVB2O3"
    "U2wT-fPa70Kz+PiHBkTUssX>fY5<X!Ce!6j%~b<lT9gmdsb~3VF`j0YAC?flF<D}{?206vAWhsPX?ffPNfGQHpP!$L@nrhz^YaGkd6U+}gMI$|"
    "aJ|}}e<QU!H$2$k=_#om;W?mI`L5x?nyUt$Rsmf#-~|jz1fxR`Nf!((zFsx3*y9P?TI_#v)j%??>P2Q@P?;5l#`hz{z)Bienbvu{J`e}Diirn<"
    "A!@H0;7!+;s|J$GRRef`Oq~_m1f%r5h6nq4)qu-2Ta#6-rG+|*cA<qjEiI&fOA8@nRaN9P&F+@_25(vJA%e5b?Hv^ai();U!VDGD*iQmiG`kzQ"
    "O3QMaf3n==wqdzTDutA<yW6x~uif3ma{qQ=xhcjbmiySS+(h-*1{168VVycQU!7jQynH^Ux9^@`J|q2q_YASSTOi<9OfHt&!-APZaex2i!{O=R"
    "-Xi{cIu*B%`%|&J%@cjl$L_ACTo?l>T&-TF0YVdExe-s)D+;tcUnok;ax0qLYgukx(;Ajr*NCdssK_G2oW_|+wRMNCCUAfqT}@z<0+5!jDl)O$"
    "3J$p<mOIdJ2oHP@$ii^6@M?C;at|~=9A7UgZ1f;0tJf@?0<Cs;BhJcJ``u>l5wEeCmwuEH)&07=O`aFK`{x6(+||v2Sng{5J<GjsS?*=aaxZU="
    "KVrE77_r=s_wOz&x7o7X;w#H-)9$tf#|9pDTb5gs<%Z>knhRaVLni2m<u+8cVYx{$Q^iB+w%nLH9EJ@y24!h79x}1qK3zM9ppt9!C0DIwx&3&2"
    "-LTw2T8*Ze5n#(0+}HC}!*Yi$%N@S5+`9|QUEMS+ceVeH<=(d}_wJSDPBGj2d#L+-l}1?_{DtL45NR03;&i(6b*P5r)|6qxa;w9|jjzji$Uzau"
    "-?QA+c##<)4YgspPv5ZI>r=yW?;DSPw~4D+j-3?7p6`uM-?H2)6&L?xxzlf0?)WDx_l2D8eoM|yh@7>5Le9!W&PM-0&N3Jn-Gb|izLK+V|0i<x"
    "kC%o++&%p9@V12d2|4>xU&z@yX~^05PjZ(1h@9m_&hiU6d-au^y=uu><3i3FdX&;#whT?ZYROrt)495kvr?zAc0<VzE?gshvB&HY8+0LOJ6H6b"
    "S-IcQ>DEGYcOhB-lbnrzNX}Xpa@P81a&}VR{)n7Kq<0}_KmLH6Ez4JO_WW~l_WVO~wkut|lC#pE<gD6r=muI>7jjnWwB)Qr<ZQz*zPXUI(p5vw"
    "4m%#V6v<EsjfR}P`lTUfdv7$l?H?Xj-;=Y9$XRwFXK%lfv!oVbOU~YYC1+zIXB!|6AD`ZpEjbHSZ_8v<L_Z^EZ~sSfRw4$qOZ;F<&UTs$k{w-="
    "8giBqIm=$j**{14kL0XrD__akK9RF+pMN7~O}Z<QrVi*;!<N;7G#J1k8|r_qI-Nn}!+CT%iW*$|z9kJ!&-0lC=f;Q)(;Y;S*402$RDcRUq#1g6"
    "oyL}KI355a<!gqm81VSko0eKcBtq08dl`=%+eMBvPN1@cs71tRvZOim%@8c}@2JJ{I%aWH#tpRye`Uiu<q7Ob*iws$bD<We4YfGQ6UUNmtnWN^"
    "ES*RB=JviGIc<~>P?QiLN^n#}5}}_ajwbUc+T2d-^l!(&bU8L2YV$Q}(cDsqLLM|c2I08czkYcm(d7Wrzu)xw3fHvZFLbv`jTlA+V)*9j%>d>>"
    "SO0yb$X0-TdNrh-Ytc<&jE<rMWFJAcMRzK5s0PqcT^B)>q`M5lmK3_#h}G#T2BX@dBs(rPb60g5(Z!l@O+|q-ui?NH)pDaqKs92Rat8}3nlG1m"
    "Vnw47-GPUc=OrF}`89~+>&Y}v><|h0Wa5+Z{lq8eFj3)q@Ue8kmCvb8Ov!*QGa&>Qt_@=hv*|dn$#l~m8_UPGEyvV{7Jb>(kS&LVA<Hy3%)-8E"
    "=hmUQb#%L<8E%SToES%>E=1!sjAQ$(rZvwJALk-N8JIDi#L}WtPNOu{86xFD;^Me2ltZ_!F)TR>oG6WwC}dFxfiW7$1O74#dV&Toe3}6Uk8@0S"
    "QgY}nOnAr`D)LHVK^-~`_``SA%UH0neA;Y+?xh4gFc`Qv$Dn{3$k07>Y}G{9mWhY<wa>4e30KveK7F7q8?I@n%_jnr{t%lB^OUf;%rb9zdp0^v"
    "a-kb;G09`ybhB79U6msS7Z-jk5Dh<8y{uM<8=`XA@MFt{9|MS1UvvKle$2A}%8!Y^^JBW>t#;d$=h_~n&A%P@qHNOUQHiv9RMu5&Q1HE65<e!^"
    "yUp9pZY|2~?y%dI<@OpPFS)EaN$Q8(=sRgWz1<*}30i(ExbS1d+jo5*CX>FR=~rl8^$@fAR%f6x5_I(v_Nl2}brd6BM1d_0ExPqlSFwBu6VV5K"
    "ufxXXVxvQm3rQs&I_l~yi-^Rwmy5+Dq?<%8wySB<m)T@BOZqBnM1QwhO(O+{e2G?KGrqXLPoW^N>sC1?aCbhuf84IZy5YyF`rZ7o8r5Zz2VT6n"
    "yLsF#r{4H@+%K2K_&8Y>_y>F7Ef1@PA45)cQxr+OCXohA6VuJK;m5+39}9^e3n|Wss-)q^+?F46U-_|)1_YzX?GQf}GrAC%7;#AxhvRmgN-AG("
    "*BO1SkVH@1KOFWWO%p^z;Y6LrSgFJ1787&?r*DQioGnw^c6BPmrw2r(_uuhj_3;4<U|jE}%kdb`<#Dp7SrB_wq?iTwlbih<@o;ieL}45W#<PYW"
    ";~5EzSrswDJINFzxP%n#Cc`k+nPQ4Q4GgRHG-j4i*7GW!X5a8*SsKS4r>+sGhHz(1pIq1e>@3rDG00M8F$Li%+#{#!q$P7ech~T|{t$a+etqGT"
    "=U?_aq+C(G-z}iF3*U=iC4?oL4c8;M<=$WROx5n0iuO!(yUd7_5-D*#2u0F|&tJa0JgvMZ+B46$PrpAuKinT5ZahkT={ubsv%SOfZ|C2h4<3>D"
    "&&RqxZjSZ19>>7jTWxnM&(X#H_#1IO-l*Yv%F+G?2~%Chy5*!f_KYn%qx$}KlmC6sNWE@93{Aw0y`kdQacI(AbmPz_MWUw+%_2iCp(vU|T#pMs"
    "k=9v^FgPY5D^^*+7|``X<I}Y8bc2=CBGD~YB%0+a40~Y~!{zvg55F#RMPtif*M*|9Mkt5FZryOohx1{ZH=Oc)fkX`g0nrY7COV(b?;j7?Gv{{C"
    "oWI#KclRG3&%ds`am)3LiR-EABEg;^zHK)xI6}^b=4rVe><F>h9QUg_qKkv(H4Ohehe&*8SJmu#9AnQ^v}X+3Glt6yo$k?8=aJjf>E@1%Zu$t}"
    "_6(o6o?6lQ;dx(UF*ZV`XOFd^i@WDj4cBq1H7v~oF?x>gtw?!%HA9ADHhbo&-7~w(p4nrwe10abr#>&1*W+1{PtvmGdddse6Z=Wa^&~{d#ECC>"
    "J;t7~cuvB%@aZu!ot`B~@SZqeNEpVxRxL54fzh6+Y0tcJJ+x=O<$At$E%G$6cm!wJcGeZ!oc2tSGMgz$jIBYh>a1%I6uNg9Tp0GkB7y^fe)#X>"
    "W0Y?XhsVc5zTH7=mMJzC7cnjKSnzoJ@yma|eB8z#U*JbxKK}9Y>E(|<KK=1Nz+614lBR+IxR)<qK0l<l&s0ZwdU$?@INsk_ueBE25<jc8=ng~H"
    "hJ~#<Nt@h--z0a_d@@a>zA}vH()B*Ab&5L`#dPv<tY~Hw(QQR#!<tsYHf6)HBKS_>I9@PYji6EpffYNy_7Qb35u{&sJ0KTii`_(sXj(&XP97p5"
    "GjjNH-c8Bpp7(o1P`kbFC8D7(R9CPA5L*i2FMt@JeLQfy`LLdQ^TYYS|NZZISD(KCurFr-@}+@neH~oi5QrbP34!?O(+7YE@#(|+01%!(efe@a"
    "M1bzo`R(1qGl{#q8$(QwpT4{t7r|n8{Dho%vAf%DCfjLlC$sl}sp8q<=KakihML55HMFV+-pgj=shSs!Hxt$H#s!4Rg2T&sGeyk4Jny!S6E?!e"
    "On>|B9+O``|Mmp+K!P{)MGyq*k5C>?zb9gj3B6#~9?qoJyR$d1JskJaPO_OM{X1R1{rvgU`E-2vOyYQVbN%-B-#?uYIsE<>V!ND7@9sa`Ez9X0"
    "iD?<zQCz>hQSGQMRVPxTHI{~Lr^k1;liK*>DGJhur$D6a!>14NdcFM+UnjxoSbsi29Ixvob-jGMSx#==-EGS0hWczfLoWz~tixRxM0`^(uB(z>"
    "4b{NlbVD@}E#MZAVMmXT|9wEyq<sKecX-@w^W_GKhhuTk)c8&LQU$!fQ~@ih0#?i0D$_2NLRahmv{H!lpT0rlXwuixM0{ORX(BE!rHQ`cy2!k="
    "f#Um4Mp={Y`&n`w>0Bg{&RpH{)AfWqnuYhwzU=or%F0>2A4Anh>k}`_cjV=q(KuZ<UT!>I&+lufd0myG3>$wriP@sv_!HXrPfxV*pPG%2MZQjX"
    "V;0`OU(&1Z-_xrQ>t%Y#Q+N6PeFGGyR|u&01k^fDGvA%GATAj7Nv}O!K(rKsWu1;kPJKULkH=L#hZvPr*5~{Eml={mYrL;#WfqNVh$Ia?f+=%+"
    "Ux}`h3;|=5AYet1qz56ZUv6)O8HDxdHaGoX1mT&Z>cf}w+uJM~)7NUCAktAX$Ft{KAV4Go6a;{HpamUI=eu*G4rhqy;lqcz{&3hGKGfwd-<2Ue"
    "(!63Xum|&Ld_F+n5sgB+vH=q~oiDQp=<DO-EIYk}7^PW~-;U}qF!NF1o1Sl)SytDl+)Uu9%s4W?AK~tNuCuZ{*D*q@-D6!Yvt=2Cff#4?xda?V"
    "9DdVvohD`!n_oLC?&>HB{UY_b&yp~Se0r7p2G<RE0x!Crb72ho(T>O`Ul@zszDAg|*ix+I`C<T5cqw5AtJO=@mavy4m9UqW61HhOZO!bBGakP%"
    "s+lF@g6=GnWS5$m!2=IjxT*3A>HPayV#gDhr$q(@0Y4sG&o@(H>%rm(D|noev;TD5EjHV`-D0|(<m>fr1sfe5PG7!!I<LIDx93j?zo4FB5g%?4"
    "on2Q^vi(3F>E`;wr%&XOpkDs?0FU&5(-hUa#ta|ke>KD0wT4S9(E~%uhJC((3phMRFam;eW++pvg#*ofzdzHo_qznvA9I)c{%-drrSC~ic8d%N"
    "T0KRwiU%T802WzD`9`Z*;5p7Hg@Es|2_{EVlzu$U9c1-Ubh!&0mi5aY_fxN2-hH^gn?gZMr!hIl-!_}g?)2&R-;rm*sx7x~Cm8&1=gskS{`C3t"
    "Td2<ucZfs(`{f757v|Z+S|xSftr3^bJr8#LkC%^YD0px$h=qrk{pq*oHAesRaG8a=-JO>>rQ2Nyw}8P<Hsi?C)MD%zDjw?c_~Gf@O}V)vfuYg}"
    "m(2(_Zzmr<f3A<G)93mcG4OgFf7n7K*KrU<PY-Djr6OSVyCc$&)UoZdj%+n@R27|-WQ`BU`g&1UX%qzrhY=@O^eVLtbzloGFHJ=oNFtw{M$FK("
    "IG+@*p~sm^U)a=ClOb196}rb<oL+}I-4-&&Suk{;gl%ZViTFv3{xmB4&#KBch!d$?sKhm3Slu99&pg|7qk6T6s})gF&o`SeO7eN=Cn3vo#PiJL"
    "#dJqRuSZlTUE~c_gn8lb4yz&yz16Sh$B*}Sam#p)iSa~4SS5#Di5zEjczln5IjV>gIoH+W{9KpyC}Eb_hp{~$h!ZKA?@8OmJr_3Jl3f{g^X0D}"
    "4!)`=#roGJ)1c@|KwVLko>xqfVPVcmF)a}dCk3-~*U}WpCB~ChV4T&dr7Cf38C0^B%`mQgW!UT4VN5^wVUxh4=cfRtu3algt8vYta&GzDn5^zk"
    "$6eGio(b`Z3E&j_`={sUyk$IJiNzv*{D|d1jOR1da&fMU*%+SSDZQpZnpD2Ka3ap^gcJ*E#XY(p&@_zt^n53DsODk;48Gn!AGi)BOoi?M)5Q4n"
    "#aL$aT%*?`#<LbylIFsSFx2hon-PA;c-~zYPvS(dgHKH{g(*6^wlra>L&VQkk6Y;T(pj-hyeJmU9Y5&`aeaVSyJ+i!uBzL9H0X4${$u(6b)YKr"
    "?IF)IhZ5svTgj$X7a^z}HcVVnMRv=?xx~b|4HFln(+OtPFmY3u#+Hc-iHQq&6uNo7S|E#Tn7Gwp=_N&RgP<I0P33I1S}pRx_omC;<IUYBE|$yv"
    "@i;D)6NsYVyJeC1;cEN%{&2nvn^pIYR^7X1)!C}tO;)Sj601x}@9!6!+37t5m6W@kt7N*`F6f6blG$p)<bE=vd-|yTe&<T$#bLy>PA8(Gxhh2&"
    "YYX6S>>ygihJr(aD6oc-@adZ-O1Fzprt;}aR~FJCwRdGPb<m+->7gx|j8eB@;xY~e!MK*M?pOPD6o$oof0Kk!agzuHN(96<N(qR40<}*71b^H<"
    "AooVnO=>ya)mb`@miLEt)gAt1;)HPME6N;d$vBldbxj+ePIrP7Rk=)c*vZ!0HCLsqxpkA?>owIFwZoeG-RP1;Ok5^ts6t0ZRm@KqVMrqkW41nF"
    "U>OGH#WBBI$JH#GURK@sb=9fp+Ej7s`&f0R#7#+h?JQd-+>eoTQ_aI{Rvlh7Y1PrMx)`QqsnSq4ou)*Kw1;k#2-OJLavT>HzHiwseN<D^6`t3Y"
    "c5U0$ICnX>Wf)c;&wHFV#Z<a?g)gX>ng;X*eLKmglj}GR{K;fem7(8=#CO5~2Ej`5<#HZ{zPDU1N2ADVMB=%0>rN{SYp%jri70YmcSHn%$FTF}"
    "jqP~}JZ-nb=oYZD#6tQh2esS1`h{)~OSkLq*qphp+CjdiOE$3nm*u*FB`a6|rdx@c!z+V+0z}7#8)_O?`n@;*mNohjt6yYA?!6iI-n_8@h66Vy"
    "8@5Bgr(@8KJq<?|^p$f#cTlB@F%(@j!nEjzKtyqxh72AcN+YN+0*-WRb;oL|!gPHGfCuCTl530^zNQh5s$pO*KF1~H(ib_EL8o~_?YOTrPCLyj"
    "nr6q=I~|j5;Ol%HYBNI9VvW!kjoNkTzV)ukINdmxKFd*>!ZqGUv=4Pnzw6R>_yxMx&hSQzT&4Tg+cMp@vQMw-_E7D2{_{&mhbI~S>y0G!;YZ{*"
    "-F|;SKjPCL3?Q!FbQv>E^_MGRMX+72h!w$z`?8|Yz4f|6;N>@0U6?V?)1e*M!A@UkL<fPnsY>1c(Di)ZpkFu&0u!ncQe>Mdu^MTbM2KiJ`YjtM"
    "2+yW_pu==VaT14?9wyK(sm62~s!UtJp)FwAHl1vQsC35VVP7$I%|zHT&~%l4l&tw>GF9tKLp-OdL#aC$_J)IQx7+K|zs`TI-bkw4@6zwUbo+Fl"
    "MydIgGC*R{4;r<QbO;g%%$ofPghW8l_xUt#8Tu7%O{nfrCvn9slkTZuTCjn#LIO4sTi>S5X_11kWZV4fEdDn5?r_*sWDL3244MYlZEwoi{Ok9I"
    "=(^YMUw#2h!5@Pm3Hk{&soU%KI#Rczsy$h0L{AxD;3WEA`~2ElkJ_E)Q#xI#D>c7EMu=R!xf*cp*v3`oLV|VVh6Ix|%u&*ejvV@M<!?Ge1AZ}Z"
    "q@f|gkV~r32q}+U#~jF}mClg@o2C_HS!fzohRDs3r`$0)LYehuJ$2)>q+5kU-3ERdFoF0i$g0V7HsM~FPiKg@!+aL7FmP<(#$fS0!SJskgk09_"
    "ak|YNQGd3no2CilLBcR~U8j5hnwkzHtTk9SKVdh})ItAhFwktoi&wI)>4w_TbX}!8Zqw6_sdeaSQ<F6MN~@+xLld874Q0!;py=lq1}5E+nlp#q"
    "!I+B^XhW!7q#788<udF@!*UHQ<bg-`x+NRrkixchw7<{dZ-dAGG+a^x9b=a1CK|eAHbR%0zhwGrXzD%s!=gWQ`oe{-VWX?`uirjx;Ge8DKiZ~i"
    ">hNoyubt5o8od!8*;ipM1~4-{P47UeSq7)t=_)K+=Q;1nPMYUoPjfR8y*#Dw@~W0g_f(?W{?I*@ph!5%O1P$9l=Izgw{p|*m~N;8wU2@li3u}<"
    "ytrN>%Hc^>K}S%lD99qLA@0G0(dB5!ar_`?e}s-R1dA}#SP4i;8vqH%=E6ruhsSh?)RbwbVawTcry=|QN@}{2!BU(oO{4YZY8GBnEO+3-!!zAL"
    "2ZA?B(^^s!L*Urw44&D-zptcbM5Jbfd2rKXf1L$_$7b-}>)^>fAd!Sghd?B(HlHrhr%RSAiH6j?jHOS1(A{dtO%bWd|0Fd9{eTj)9QxKDqnnsp"
    "8hsI5!9#xSj9iKC_TiySSEl=%tFA?qXUFWzuS6O@9CoX;CDL*t((;B#OOa(Rce?}K@Th!zM4nyLkL$;}p4D0Ai*SLniAR`kx6AD|A8lvb(Wn@~"
    "gK|8-ye{$}g08A0T9L@;ijT=`I;qeKX0vG?dcllD$Q&z#Q;3L2gHA%-Z3Gc%iX#wnIajh1#If5^oknyNu~@IBiA3b!H0f%r5nZuduUDy}a3tfI"
    "VsH{{uw4IuNLwtIu~;nW`gE~K7l{a5`?`wa^?F%DEwd7dY~bW^d^pf7;zK(N=0}^&tVCA4ArW)N(mcsy9V#{$GFzErs9Z8Q30LBi!zPyvz9!Mb"
    "z;Zxzm%ApHm$M|8q?07T`7QJKP!>5dUb>8u?(q>LmdVj4N1tCiTjvOt`bADB6k#0>0ZLkXLi5}YcZY|Ehr=!`KYV!j@S&_ftUt7;EEIGftix`%"
    "42tG-mS0aN7S#$HcU~5LwBPT><22fnpt}URjR`Lj9^Bq;$}|pcNu)kAkw>ODWrp2&dlL%6GxdRLU`P9Q52+SBLr=3B(KTtIEmPBU7Q0n$8qO-$"
    "5!2w=!#1}pMw}EB2}fz3G&o;2%3bCL%r!knwd+RZ%iK%3JD<k%RHM==rr!!E<LK$>Fdk>g6A8q-QFJ<O(Tq<dGM~G7l~<YRHX=ln!7x<TGZ_h|"
    "%N9?kLNnQ+P)v8Yd^%OkVfU2-%<<WRF1;sPG_KO8@u*7i30aD+fkWW40)4UwHu%z+dnj;%`lAXaOUVG9Pp@i@WYN!ccGDI182tuV-ezw3YvwlS"
    "a=L>Zh2BwfM>{LRABc!8h3+IbPZ5y}!Ym6c(+o2U?u35F#rHjzety*q0{Z<)GoT-#97s|W$2oej&8(tZwv%$r(ACT778}@4GP>R|f;z>17V|kR"
    "6+au(HO>*#QxcYmPuP1@WhGUm2aiRbLw`_(nba~r{C~||*>2l76n$=MQ4~drn|3)~;w<efE$RjV1_}B&|Nl924&}s2oV3L}RS+&IQDMaG-g9V*"
    "cRB8sU?vf__6+ttEMymbohC$}rLMjYb1C%Yw|~9@Ib440_(WP{G&NRryT&V=1z+`Qz^g)RhLBcdUf>OPtBc(kry@Rn`s3f-`g(V^|2px52!6~B"
    "CHwLFz@5Lk%3s|LweM?d&=$4oYUVj-_UUd1KypSGZyg%h`;w`M16_)YHGeUH->gw*T~(|vxJjc|YSFD$f1O2epmDvPqLJg8j)=G)J5+ve3z5+U"
    "3c`9xm#QUsF_7=>ybD{XiQN#l6vN;zX$U{BSKI3^x3?Pvx0`Xh8DJ)FzO88{@9wq;?zS+K!`GXeo9*V~pL+(^7yIpMwdwXKHow1N02Ma%tEqj3"
    "7h8LAf4?6YjM0L&>HPdNX9DE;XYQuvi=u}r=_=80=~2VwIbZd1#H&KU{#}2$zvb5Kw_8%-{r%nc^6L6-y98n1|Gheg3S0Hls)M?^=P&yhCe`K5"
    "7%Q9#iYv-KO3BPPA8>WU`O9(S0y$_z)I3uW_aHF`j+l$Y5vv@)#mnd=jCf0qC>Eh5M-<i`_Ihd2f-s4#orxEMBr0x82KX?;uO;EvlJINhh%b%w"
    "%n>(<BaRzba~SiQ)U4r@@-Q@{YBc#6V?YH%kAMQ$fgF)AB~e9~;N}HK>|f!CB=<e-dlHEwE}Ir_*vuS}i<cmp!J(SS5l5Q*DFK10Z#r^B@D2R0"
    "k`W2iE}2=CG9%L@iKNMg`7fD@cqr+kIhImp?<%j$vdFZi=jT$0Fo2~so?Hc3IOrr?C?}RR!PRGwl~(5zmGNWTw&M~t;NK)L>^rVRL?%#Dv4@g!"
    "(QBhM{-X5=T2CUU3#}Xa9IX=QHnG-9c<!V}L^oq|a1LGzhim%X%wN1}Vg(M!7i1Hl8jTgvd77BKhJi!kHHed&Ch#`lGu@Nb5QZCW*D}<SzCJ@~"
    "g95y^Ht5h0C)1Nx$y6kuk2>_r5J2ESy^J~XxOS>z>rW`i)HLlWP(=kc1bM*J+^zwlb#$FA6T>uxrMpa16|$6BK&2Zg0zC!e6o?+jKw5eHVQx>s"
    "_9U;Jxi6kf<TvM{(PyvfDqDeBQdGJG#Vniyx#aqKt>|w$QN<iQ+u$0fqz`UIpU%glC=zKE1#Zs7G7)W=1Z0+r{IGuYCH*=qR<+%{s<N9`Ri>4x"
    "qV${&qt8nU!DXmG61Zm-a6g2VY0W7Y54MhbmpTybuECgZpN{{NzRdP|!MoKH-VGoQ`fqF!(t2+}z8%Fg-tE0Y8ijJYLXU#@5$~2Kyn9|uXe{|f"
    "Z|Z*Rb=|A9a!~8E#1jZ=q$oJY2$DH4h=A{jWG+z`EPUUh@cazipK7`IyOPYb*b9_|#Y_RY)!BhtO;5NLp!X&n15o&ZTTyK<liD8N=2q=fZe?HL"
    "R`zXf)je{n@fWz&_%^rV`W&uIzyJq&?Wfo{vr0eW9gk+^SF2$z<aj7#-OxcQ)M(G$b9?B|HeI_G*RJhcrRm)$Z4E^s0(*kiaXKq?l)MN57s135"
    "@owU_$T7lnbf_q(WV-X3v-b8OPy3J={t#KoVG1x7s0nB1)U1*v%0q3%pa-IJO*9>><Mhlrl0pxu#~w7j<ygkBYa4EUsHC3<a?OEUTgjf?dOY6J"
    "n;9<}UbxzobbW?mL($m+8g!sTYTWjWc~@Ix-PFp~bek||hKDsCb}+#FrJ~XWQ!#vXjxI^7kM>G}R$KE%(3*aoHO_)>=1alv-vmzaT*27mi1S@<"
    "AL|ya@mRSLT_=RIDWVBXC^cfJPF>2DpQzvz51HX%7wG_v6^%!<WGdq_(>*(%HRxWo>0t;79y$x~(9M&<zXBe{BY5aezymiYPhI&H@G!m&9)279"
    "3*~E%YTQ@e0zzP9INCDr6x6d-I2E_he6VHE;^@i}r_3!URq5vS2ewR}*s`=yo}|i$kZBh@U_Xiz1A$VoyZ=uEkv6iTk->jd+@&fV00YQ2pe}=p"
    "m79;s9gEI;dF686s9cd!m2t)>XG&E$U<bv$XwqS9@<qzSN8_0*3^YHO4K%Szp=d!EFX+TnbB5Gg8a+Fw!Z2H@>@l<BOFq`PEK2Sg$~kuc5`m(M"
    "Jd?yFxGAzc%NAKGrS-7Ph!}}eK2+YEk^+yn!5HyI2*kA%EStp1-vL}{oI+sc0A;RF{=!vxjyo0q@z2{fPn9nYa=ywle(jYla^f0s_!$A9ia#=B"
    "sig?9M7|Io7IQ2U*#{vOA`|c5y?^%~(r+(`"
)


def _hunt_table(_path=None):
    if 'T' not in _HT:
        try:
            import zlib, base64
            from itertools import combinations
            offs = [(a, b) for a in range(-5, 6) for b in range(-5, 6) if abs(a) + abs(b) <= 5]
            hb = len(offs) + 1
            org = offs.index((0, 0))
            q = np.frombuffer(zlib.decompress(base64.b85decode(_HUNT_BLOB)), dtype=np.uint8)
            tab = np.zeros(hb ** 3, dtype=np.float32)
            keys = np.array([a * hb * hb + b * hb + c for a, b, c in combinations([i for i in range(len(offs)) if i != org], 3)], dtype=np.int64)
            tab[keys] = q.astype(np.float32) / 255.0
            _HT['T'] = (tab, {o: i for i, o in enumerate(offs)}, hb)
        except Exception:
            _HT['T'] = None
    return _HT['T']


PARAMS = dict(
    H=12, S=5, gamma=0.92,
    rho=0.02, rho_early=0.05, mu=0.05, mu_early=0.3, term_k=1.0,
    tie_cost=12.0, soft_w=1.0,
    kill_c=40.0, kill_p=45.0,
    info_c=2.0, info_p=3.0,
    open="CP", planters=1, planters_late=1, planter_from=4,
    plant_ratio=1.6, plant_min=12.0, hold_p=0.8, steal_w=0.6, cell_loss_w=0.5,
    seize_min=8.0, pdisc=0.85, pp_cost=10.0, pp_stand=28.0, lunge=1.0,
    guard_w=0.5, guard_min=20.0,
    recruit_until=40, endgame=14,
    campers=4, camp_from=6, camp_min=0, camp_div=99, camp_decay=80, eval_T=30, eff_hi=0.6, eff_lo=0.35, eff_rate=2.3, reprobe=0, kill_win=30, kill_n=3, lost_max=6, probe_n=0, probe_T=12, probe_kill=2, probe_win=20, camp_orbit=2, orbit_kill=3, orbit_win=12, r2_hi=0.75, r2_lo=0.45, r2_kill=0.6, siege_hold=10, siege_break=1, sb_T=20, sb_carry=1500.0, sb_off=500.0, sb_gate=1, sb_recall=1, sb_guards=2, sb_gap=25, sb_gap_p=60, retire_entrees=2, hunt=1, hunt_min=150.0, hunt_R=5, hunt_n=3, hunt_hmax=1e9, hunt_camp=0, hunt_Rc=5, hunt_targets=3, hunt_pat=8, hunt_cool=15, hunt_fw=0.5, hunt_fcap=250.0, hunt_sw=1.0, hunt_scap=600.0, hunt_fmin=0.99, rovers=0, rover_d=3, rover_from=8, hunt_ct=1, hunt_ct_d=2, hunt_ct_r=4, hunt_gate=99, hunt_ct_solo=1, hunt_tab=1, hunt_eps=0.3, hunt_vmin=0.25, hunt_tabfile=None, camp_late=60, camp_from_late=3, recruit_post=20, shadows=2, shadow_pen=3.0, shadow_minfree=2, lone=1, lone_R=5, lone_stick=10.0, camp_occ=30.0, mark=0, mark_min=40.0, mark_R=4, mark_reach=6, mark_n=4, mark_chase=0, shadow_post_d=0, shadow_swap=1, shadow_hyst=2, shadow_from=8, shadow_start=20, shadow_trees=0, shadow_end=20, run=1, run_mode=3, run_ev=1.0, run_small=120.0, run_T2=8, run_n0=6, run_b1=150.0, run_b2=300.0, run_T=30, run_k0=0.25, run_k1=0.05, run_sg=15, p_extra=0, p_extra_trees=5, orbit_tie=40.0, sb_static=1, sb_still=3, carry_ok=100.0, def_always=0, gate=1, gate_min=60.0, gate_w=0.5, gate_cap=150.0, no_p_T=60, predict=1, siege_spawn=1, siege_T=3, siege_carry=150.0, camper_max=20.0, retire_stock=2400.0, max_units=10, retire_from=85, retire_keep=0, retire_min_left=50, early_planters=2, camp_lunge=0.0, lunge_floor=0.0, camp_end=4, camp_end_from=28, camp_planters=1, displace=1, avoid1=0.5, avoid2=0.15, sweep_w=0.3, sweep_base=10.0, sweep_max=80.0, camp_tie=40.0, camp_w=30.0, camp_keep=100.0, camp_until=0, base_def=20.0,
)


def _rc(i):
    return i // SIZE, i % SIZE


NB = []
for _i in range(N):
    _r, _c = _rc(_i)
    NB.append([((_r - 1) % SIZE) * SIZE + _c, _r * SIZE + (_c + 1) % SIZE,
               ((_r + 1) % SIZE) * SIZE + _c, _r * SIZE + (_c - 1) % SIZE])
NB8 = []
for _i in range(N):
    _r, _c = _rc(_i)
    NB8.append([((_r + a) % SIZE) * SIZE + (_c + b) % SIZE for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b])
DIST = []
for _i in range(N):
    _r, _c = _rc(_i)
    _row = []
    for _j in range(N):
        _r2, _c2 = _rc(_j)
        _dr = abs(_r - _r2); _dc = abs(_c - _c2)
        _row.append(min(_dr, SIZE - _dr) + min(_dc, SIZE - _dc))
    DIST.append(_row)
N1 = [[i] + NB[i] for i in range(N)]
N2 = [[j for j in range(N) if DIST[i][j] <= 2] for i in range(N)]
CLS = []
for _i in range(N):
    _r, _c = _rc(_i)
    CLS.append(min(_r, SIZE - 1 - _r) * 8 + min(_c, SIZE - 1 - _c))
CLS_MEMBERS = {}
for _i in range(N):
    CLS_MEMBERS.setdefault(CLS[_i], []).append(_i)
NBR = np.array(NB, dtype=np.int64)

REGEN = 1.03
CAP = 100.0
ABS = 0.0375
COLLECT = 0.3


def apply_ops(v, ops):
    for op in ops:
        k = op[0]
        if k == 0:
            if v > 0:
                for _ in range(op[1]):
                    v = round(min(v * REGEN, CAP), 3)
                    if v >= CAP:
                        break
        elif k == 1:
            v = round(max(v - v * COLLECT, 0.0), 3)
        elif k == 2:
            v = round(max(v - v * ABS * op[1], 0.0), 3)
        elif k == 3:
            v = min(v + op[1], CAP)
        elif k == 4:
            v = op[1]
    return v


class Belief:
    def __init__(self, me, bases_all):
        self.me = me
        self.val = [0.0] * N
        self.known = [False] * N
        self.hist = [[] for _ in range(N)]
        self.pure = [True] * N
        self.cls_v0 = {}
        self.base_cells = set(bases_all)
        for b in bases_all:
            self.known[b] = True
        self.cls_v0[CLS[bases_all[0]]] = 0
        self.prev_workers = {}
        self.prev_trees = {}
        self.nz_found = 0
        self.cls_seen = 0

    def _op(self, i, op):
        if self.known[i]:
            self.val[i] = apply_ops(self.val[i], (op,))
            if op[0] == 4:
                self.pure[i] = False
        else:
            h = self.hist[i]
            if op[0] == 4:
                self.known[i] = True
                self.val[i] = op[1]
                self.pure[i] = False
                self.hist[i] = []
            elif op[0] == 0 and h and h[-1][0] == 0:
                h[-1] = (0, h[-1][1] + op[1])
            else:
                h.append(op)

    def _learn_class(self, cls, v0):
        if cls in self.cls_v0:
            return
        self.cls_v0[cls] = v0
        self.cls_seen += 1
        if v0 > 0:
            self.nz_found += 1
        for m in CLS_MEMBERS[cls]:
            if not self.known[m]:
                self.val[m] = apply_ops(float(v0), self.hist[m])
                self.known[m] = True
                self.hist[m] = []

    def _infer_v0(self, i, v_obs, ops):
        if not self.pure[i]:
            return None
        if v_obs <= 0:
            return 0
        best = None; bd = 1e9
        for v0 in range(0, 51):
            d = abs(apply_ops(float(v0), ops) - v_obs)
            if d < bd - 1e-9:
                bd = d; best = v0
        if bd > 0.05 * max(1.0, v_obs):
            return None
        return best

    def observe(self, i, v):
        v = float(v)
        if self.known[i]:
            self.val[i] = v
            return
        cls = CLS[i]
        v0 = self._infer_v0(i, v, self.hist[i]) if cls not in self.cls_v0 else None
        self.known[i] = True
        self.val[i] = v
        self.hist[i] = []
        if v0 is not None:
            self._learn_class(cls, v0)

    def prior(self):
        rem = 63 - self.cls_seen
        if rem <= 0:
            return 0.0
        p = (28.5 - self.nz_found) / rem
        return max(0.0, min(1.0, p)) * 22.0

    def estimate(self):
        """Per-cell expected carbon (exact where known)."""
        pr = self.prior()
        out = [0.0] * N
        for i in range(N):
            if self.known[i]:
                out[i] = self.val[i]
            else:
                out[i] = apply_ops(pr, self.hist[i])
        return out

    def update(self, workers, trees, visible, carbon, first, my_dest):
        if not first:
            pw = self.prev_workers; pt = self.prev_trees
            me = self.me
            stay_cells = set(); stay_coll = {}
            for wid, w in workers.items():
                o = pw.get(wid)
                if o is None or o[0] == w[0]:
                    stay_cells.add(w[0])
                    if w[2] == 'C':
                        stay_coll[w[0]] = wid
            tree_cells = {t[0] for t in trees.values()}
            noregen = set()
            for tid, t in pt.items():
                if tid not in trees:
                    self._op(t[0], (4, 30.0))
                    noregen.add(t[0])
            dead = [(wid, w) for wid, w in pw.items() if wid not in workers]
            death_near = set()
            if dead:
                groups = {}
                mine_dead_cells = set()
                for wid, w in dead:
                    for z in N1[w[0]]:
                        death_near.add(z)
                    if w[3] == me:
                        z = my_dest.get(wid, w[0])
                        groups.setdefault(z, []).append(w)
                        mine_dead_cells.add(z)
                for wid, w in dead:
                    if w[3] == me:
                        continue
                    cands = N1[w[0]]
                    z = None
                    for q in cands:
                        if q in mine_dead_cells:
                            z = q; break
                    if z is None:
                        bestk = -1
                        for q in cands:
                            for wid2, w2 in workers.items():
                                if w2[0] == q and w2[2] == 'C' and w2[3] == me:
                                    o2 = pw.get(wid2)
                                    k = 1
                                    if o2 is not None and w[1] > 0 and w2[1] - o2[1] >= w[1] - 1e-6:
                                        k = 2
                                    if k > bestk:
                                        bestk = k; z = q
                    if z is None:
                        for q in cands:
                            if q in self.base_cells:
                                z = q; break
                    if z is None:
                        z = w[0]
                    groups.setdefault(z, []).append(w)
                for z, lst in groups.items():
                    noregen.add(z)
                    winner = False
                    for w2 in workers.values():
                        if w2[0] == z and w2[2] == 'C':
                            winner = True; break
                    if not winner and z not in tree_cells and z not in self.base_cells:
                        tot = sum(w[1] for w in lst)
                        if tot > 0:
                            self._op(z, (3, tot))
            for z, wid in stay_coll.items():
                if z in tree_cells or z in self.base_cells:
                    continue
                o = pw.get(wid)
                if o is None:
                    continue
                w = workers[wid]
                if not self.known[z] and w[3] != me and z not in death_near:
                    ntax = 0
                    for tid, t in trees.items():
                        if t[2] != w[3] and tid in pt and (t[0] == z or z in NB8[t[0]]):
                            ntax += 1
                    if ntax < 10:
                        h = w[1] / (1.0 - 0.1 * ntax) - o[1]
                        before = h / COLLECT
                        if before < 0.02:
                            before = 0.0
                        cls = CLS[z]
                        if cls not in self.cls_v0:
                            v0 = self._infer_v0(z, before, self.hist[z])
                            if v0 is not None:
                                self._learn_class(cls, v0)
                        if not self.known[z]:
                            self.known[z] = True
                            self.val[z] = before
                            self.hist[z] = []
                self._op(z, (1,))
            newborn = set()
            for tid, t in trees.items():
                if tid not in pt:
                    newborn.add(tid)
                    self._op(t[0], (4, 0.0))
            absn = {}
            adj_tree = set()
            for tid, t in trees.items():
                for n in NB[t[0]]:
                    adj_tree.add(n)
                    if tid in newborn or n in tree_cells or n in stay_coll:
                        continue
                    absn[n] = absn.get(n, 0) + 1
            for n, k in absn.items():
                self._op(n, (2, k))
            for i in range(N):
                if i in stay_cells or i in noregen or i in tree_cells or i in adj_tree:
                    continue
                self._op(i, (0, 1))
        for i in visible:
            self.observe(i, carbon[i])
        self.prev_workers = dict(workers)
        self.prev_trees = dict(trees)


class Bot:
    def __init__(self):
        self.P = dict(PARAMS)
        self.belief = None
        self.last_dest = {}
        self.stat = {}          # adaptive opponent statistics: key -> [stay, total]
        self.watch = []         # (enemy wid, pos, key) observed under threat last turn
        self.watch2 = []
        self.prev_pos = {}
        self.camp = {}
        self.camp_lost = 0
        self.camp_loss_t = 0
        self.ncamp_mid = 0
        self.en_deliv = 0.0
        self.ehist = {}
        self.myh_hist = []
        self.myn_hist = []
        self.orbit_on = False
        self.ecar = 0.0
        self.no_p_until = -1
        self.siege_cnt = 0
        self.sb_on = False
        self.sb_cnt = 0
        self.guard = {}
        self.last_deliv = 0
        self.run_pend = {}
        self.run_ok = 0
        self.run_fail = 0
        self.run_fail_t = 0
        self.run_cnt = 0
        self.run_b = [[0, 0], [0, 0], [0, 0]]
        self.run_last = -99
        self.hunt_t = {}
        self.hunt_cd = {}
        self.rover = {}
        self.shadow = {}
        self.sh_seen = False
        self.en_planted = 0
        self.sh_turns = 0
        self.sh_adj = 0
        self.hs = dict(eps=0, boxed=0, strikes=0, hits=0, hit_cargo=0.0, form=0, giveup=0, hturns=0)
        self.h_pend = {}
        self.h_act = {}
        self.h_boxed = set()
        self.hunt_strikes = 0
        self.hunt_forms = 0
        self.probing = False
        self.probe_on = 0
        self.block_ok = False
        self.eval_bad = 0
        self.deliv_hist = []
        self.ec_hist = []
        self.camp_deaths = []
        self.camp_level = None
        self.eval_on = 0
        self.eval_d0 = 0.0
        self.eval_lost0 = 0
        self.eval_ec = 0.0

    # ------------------------------------------------------------------
    def flow(self, wnd):
        """Opponent deliveries per collector-turn relative to my own harvest per harvester-turn."""
        e = sum(self.deliv_hist[-wnd:]) / max(1.0, sum(self.ec_hist[-wnd:]))
        m = sum(self.myh_hist[-wnd:]) / max(1.0, sum(self.myn_hist[-wnd:]))
        return e / max(0.5, m)

    def pstay(self, key, prior_s, prior_n):
        s, n = self.stat.get(key, (0, 0))
        return (s + prior_s) / (n + prior_n)

    def act(self, obs, conf):
        P = self.P
        step = obs['step']
        me = obs['player']
        players = obs['players']
        episode = conf.get('episodeSteps', 300) if hasattr(conf, 'get') else 300
        R = episode - 1 - step          # moves left including this one
        workers = {}; trees = {}
        bases = [[], []]
        for pid in range(2):
            pl = players[pid]
            for bid, pos in pl[1].items():
                bases[pid].append((bid, pos))
            for wid, w in pl[2].items():
                workers[wid] = (w[0], w[1], 'C' if w[2] == 'COLLECTOR' else 'P', pid)
            for tid, t in pl[3].items():
                trees[tid] = (t[0], t[1], pid)
        cash = players[me][0]
        my_bases = [p for _, p in bases[me]]
        en_bases = [p for _, p in bases[1 - me]]
        base_cells = set(my_bases) | set(en_bases)
        visible = []
        for wid, w in workers.items():
            if w[3] == me:
                if w[2] == 'C':
                    visible.append(w[0])
                else:
                    visible.extend(NB[w[0]])
        if self.belief is None or step == 0:
            self.belief = Belief(me, my_bases + en_bases)
            self.stat = {}; self.watch = []; self.watch2 = []; self.last_dest = {}; self.camp = {}; self.camp_lost = 0; self.camp_loss_t = 0; self.ncamp_mid = 0; self.en_deliv = 0.0; self.ehist = {}; self.orbit_on = False; self.ecar = 0.0; self.no_p_until = -1; self.siege_cnt = 0; self.sb_on = False; self.sb_cnt = 0; self.guard = {}; self.last_deliv = 0; self.run_pend = {}; self.run_ok = 0; self.run_fail = 0; self.run_fail_t = 0; self.run_cnt = 0; self.run_b = [[0, 0], [0, 0], [0, 0]]; self.run_last = -99; self.hunt_t = {}; self.hunt_cd = {}; self.rover = {}; self.shadow = {}; self.sh_seen = False; self.en_planted = 0; self.sh_turns = 0; self.sh_adj = 0; self.hs = dict(eps=0, boxed=0, strikes=0, hits=0, hit_cargo=0.0, form=0, giveup=0, hturns=0); self.h_pend = {}; self.h_act = {}; self.h_boxed = set(); self.hunt_strikes = 0; self.hunt_forms = 0; self.probing = False; self.probe_on = 0; self.block_ok = False; self.eval_bad = 0; self.deliv_hist = []; self.ec_hist = []; self.camp_deaths = []; self.camp_level = None; self.eval_on = 0; self.eval_d0 = 0.0; self.eval_lost0 = 0; self.eval_ec = 0.0
        bel = self.belief
        # --- learn opponent habits from last turn's watch list
        for wid, pos, key in self.watch:
            w = workers.get(wid)
            s = self.stat.setdefault(key, [0, 0])
            s[1] += 1
            if w is None or w[0] == pos:
                s[0] += 1
        self.watch = []
        for wid, plist, key in self.watch2:
            w = workers.get(wid)
            s_ = self.stat.setdefault(key, [0, 0])
            s_[1] += 1
            if w is None:
                s_[0] += 1
            elif key == 'G2':
                if w[0] in plist:
                    s_[0] += 1
            elif any(DIST[w[0]][x] <= 1 for x in plist):
                s_[0] += 1
        self.watch2 = []
        en_base_now = set(en_bases)
        dl = 0.0
        for wid, w in workers.items():
            if w[3] != me and w[2] == 'C' and w[1] == 0 and w[0] in en_base_now:
                o = bel.prev_workers.get(wid)
                if o is not None and o[1] > 0:
                    dl += o[1]
        self.en_deliv += dl
        for wid, rc_ in self.run_pend.items():
            b_ = 0 if rc_ < P['run_b1'] else (1 if rc_ < P['run_b2'] else 2)
            if wid in workers:
                self.run_ok += 1
                self.run_b[b_][0] += 1
            else:
                self.run_fail += 1
                self.run_fail_t = step
                self.run_b[b_][1] += 1
        self.run_pend = {}
        for wid, w in workers.items():
            if w[3] == me and w[2] == 'C' and w[1] == 0 and w[0] in my_bases:
                o = bel.prev_workers.get(wid)
                if o is not None and o[1] > 0:
                    self.last_deliv = step
        self.deliv_hist.append(dl)
        hv = 0.0; hn = 0
        for wid, w in workers.items():
            if w[3] == me and w[2] == 'C' and wid not in self.camp:
                hn += 1
                o = bel.prev_workers.get(wid)
                if o is not None and o[0] == w[0] and w[1] > o[1]:
                    hv += w[1] - o[1]
        self.myh_hist.append(hv)
        self.myn_hist.append(hn)
        self.ec_hist.append(sum(1 for w in workers.values() if w[3] != me and w[2] == 'C'))
        bel.update(workers, trees, visible, obs['carbon'], step == 0, self.last_dest)
        est = bel.estimate()

        # --- unit lists and maps
        myC = []; myP = []; enC = []; enP = []
        occ = {}
        for wid, w in workers.items():
            occ[w[0]] = wid
            if w[3] == me:
                (myC if w[2] == 'C' else myP).append(wid)
            else:
                (enC if w[2] == 'C' else enP).append(wid)
        nwork = len(myC) + len(myP)
        tree_at = {}
        for tid, t in trees.items():
            tree_at[t[0]] = tid
        adj_cnt = [0] * N; adj_mine = [0] * N
        tax_me = [0] * N        # enemy trees taxing my collectors at this cell
        for tid, t in trees.items():
            for n in NB[t[0]]:
                adj_cnt[n] += 1
                if t[2] == me:
                    adj_mine[n] += 1
            if t[2] != me:
                tax_me[t[0]] += 1
                for n in NB8[t[0]]:
                    tax_me[n] += 1
        n_alive_trees = sum(1 for t in trees.values() if t[1] < 50)
        my_tree_n = sum(1 for t in trees.values() if t[2] == me)
        price = 20.0 if my_tree_n == 0 else 20.0 + 5.0 * 1.235 ** n_alive_trees

        # enemy collector reach maps
        minreach = [INF] * N        # lightest enemy collector able to reach the cell next turn
        cnt2 = {}                   # cell -> list of enemy collector cargos within distance 2
        for wid in enC:
            w = workers[wid]
            for z in N1[w[0]]:
                if w[1] < minreach[z]:
                    minreach[z] = w[1]
            for z in N2[w[0]]:
                cnt2.setdefault(z, []).append(w[1])
        en_p_reach = set()
        en_p_stand = set()
        for wid in enP:
            en_p_reach.update(N1[workers[wid][0]])
            en_p_stand.add(workers[wid][0])
        anyreach = [m < INF for m in minreach]

        # --- carbon dynamics for planning
        H = P['H']; S1 = P['S'] + 1; gamma = P['gamma']
        endgame = R <= P['endgame']
        Hh = max(1, min(H, R - 1 if (endgame and R >= 3) else R))
        val = np.array(est, dtype=np.float64)
        g = np.full(N, REGEN, dtype=np.float64)
        mask = np.ones(N, dtype=np.float64)
        for i in range(N):
            if adj_cnt[i]:
                g[i] = 1.0 - ABS * adj_cnt[i]
        for i in tree_at:
            mask[i] = 0.0
        for i in base_cells:
            mask[i] = 0.0
        # cells an enemy collector is sitting on are half as attractive
        en_occ = [(workers[wid][0], workers[wid][1]) for wid in enC]
        if not P['displace']:
            for z_, c_ in en_occ:
                mask[z_] *= 0.5
        bg = np.empty((Hh, N), dtype=np.float64)
        cur = val * mask
        capm = CAP * mask
        for t in range(Hh):
            bg[t] = COLLECT * cur
            cur = np.minimum(cur * g, capm)
        pw = np.array([0.7 ** s for s in range(S1)], dtype=np.float64)
        shift = np.array(list(range(1, S1)) + [S1 - 1], dtype=np.int64)
        info = np.zeros(N, dtype=np.float64)
        if bel.cls_seen < 63:
            for i in range(N):
                if not bel.known[i]:
                    info[i] = 1.0
        dbase = np.array([min(DIST[i][b] for b in my_bases) for i in range(N)], dtype=np.float64)
        mb = np.array(my_bases, dtype=np.int64)

        def run_dp(G, lam, C0, C1, Cr, base_val, term, want_policy):
            """Backward induction. Returns V at tau=1 (S1,N) and optional policy."""
            V = term
            pol = [] if want_policy else None
            for t in range(Hh - 1, 0, -1):
                C = C1 if t == 1 else Cr
                arr = gamma * V[0] - C
                mv = arr[NBR].max(axis=1) - lam
                st = G[t] + gamma * V[shift] - C - lam
                Vn = np.maximum(st, mv)
                if base_val is not None:
                    Vn[:, mb] = base_val
                if want_policy:
                    pol.append((st > mv, arr[NBR].argmax(axis=1)))
                V = Vn
            if want_policy:
                pol.reverse()
            return V, pol

        zeros_sn = np.zeros((S1, N), dtype=np.float64)
        zeroN = np.zeros(N, dtype=np.float64)
        Gfree = bg[:, None, :] * pw[None, :, None]
        # value of an empty collector standing at each cell (used for home bases)
        if not endgame:
            V1f, _ = run_dp(Gfree, 0.0, zeroN, zeroN, zeroN, None, zeros_sn, False)
            arr = gamma * V1f[0]
            Wfree = np.maximum(Gfree[0][0] + gamma * V1f[1], arr[NBR].max(axis=1))
        else:
            Wfree = zeroN

        early = nwork < 10 and R > 60
        rho = P['rho_early'] if early else P['rho']
        mu = P['mu_early'] if early else P['mu']

        Q = {}          # wid -> list of 5 action values
        claim = np.ones(N, dtype=np.float64)
        en_base_set = set(en_bases)
        my_base_set = set(my_bases)
        en_tree_cells = {t[0] for t in trees.values() if t[2] != me}
        my_tree_cells = {t[0]: tid for tid, t in trees.items() if t[2] == me}

        blockading = (self.camp_level is None or self.camp_level > 0) and not P['def_always']
        en_carry_avg = sum(workers[w][1] for w in enC) / max(1, len(enC))
        self.ecar = 0.9 * self.ecar + 0.1 * en_carry_avg
        # enemy collectors' movement history -> next-cell prediction for periodic movers
        pred_next = {}
        for ewid in enC:
            h = self.ehist.setdefault(ewid, [])
            h.append(workers[ewid][0])
            if len(h) > 26:
                del h[0]
            if P['predict'] and (not blockading or self.sb_on):
                n_ = len(h)
                for per in (1, 8, 4, 2, 6, 12):
                    if n_ >= per + 2 and h[-1] == h[-1 - per] and h[-2] == h[-2 - per]:
                        pred_next[ewid] = h[-per] if per > 1 else h[-1]
                        break
        for k_ in [k_ for k_ in self.ehist if k_ not in workers]:
            del self.ehist[k_]
        # siege: light enemy collectors loitering at my bases while my cargo cannot get home
        siege_bases = set()
        for ewid in enC:
            ew = workers[ewid]
            if ew[1] <= P['camper_max']:
                for b in my_bases:
                    if DIST[ew[0]][b] <= 2:
                        siege_bases.add(b)
        self.siege_cnt = self.siege_cnt + 1 if siege_bases else 0
        gate_out = {}      # besieged base -> list of (outer cell, approach cell)
        if self.siege_cnt >= P['siege_T']:
            for b in siege_bases:
                r_, c_ = divmod(b, SIZE)
                lst = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    j_ = ((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE
                    x_ = ((r_ + 2 * dr) % SIZE) * SIZE + (c_ + 2 * dc) % SIZE
                    lst.append((x_, j_))
                gate_out[b] = lst
        need_sweeper = False
        n_en_trees = sum(1 for t in trees.values() if t[2] != me)
        # planters earn their slot while the map is rich or the opponent keeps trees worth seizing
        keep_planters = sum(est) >= P['retire_stock'] or n_en_trees > P['retire_entrees']
        carried_all = sum(workers[w][1] for w in myC)
        # blockers that stand still can be rammed; circling ones are not worth chasing
        still = set()
        for ewid in enC:
            ew = workers[ewid]
            if ew[1] <= P['camper_max'] and siege_bases and min(DIST[ew[0]][b] for b in siege_bases) <= 2:
                h = self.ehist.get(ewid, ())
                if not P['sb_static'] or sum(1 for i_ in range(max(1, len(h) - 12), len(h)) if h[i_] == h[i_ - 1]) >= P['sb_still']:
                    still.add(ewid)
        # calling the bluff: blockers that stand still often never strike; after a long drought the
        # lightest loaded collector walks in, and the outcome decides what the others do
        run_all = False; runner = None; run_cells = set(); still_pos = set()
        if P['run'] and still and self.siege_cnt >= P['run_sg'] and R > 2:
            if P['run_mode'] == 0 and self.run_ok > 3 * self.run_fail:
                run_all = True
            elif P['run_mode'] == 2 and self.run_ok >= 1:
                run_all = True      # everyone may enter a covered gate cell, but only at the price of the observed strike rate
            if P['run_mode'] == 3:
                # blockers tend to strike only at rich prey: learn the strike rate per cargo class, light loads first
                run_all = True
                ld = [w for w in myC if 0 < workers[w][1] <= P['run_small'] and min(DIST[workers[w][0]][b] for b in siege_bases) <= 3]
                if ld and sum(self.run_b[0]) < P['run_n0'] and step - self.run_last >= P['run_T2'] * (1 + self.run_b[0][1]):
                    runner = min(ld, key=lambda w: workers[w][1])
            if run_all and P['run_mode'] == 0:
                pass
            elif runner is None and step - max(self.last_deliv, self.run_fail_t) >= P['run_T'] * (1 + self.run_fail):
                ld = [w for w in myC if workers[w][1] > 0]
                if ld:
                    runner = min(ld, key=lambda w: (min(DIST[workers[w][0]][b] for b in siege_bases) > 3, workers[w][1]))
            if run_all or runner is not None:
                for b in siege_bases:
                    for j_ in NB[b]:
                        run_cells.add(j_)
                still_pos = {workers[ew][0] for ew in still}
        if P['siege_break']:
            # deadlock: my bases are blocked and the cargo keeps piling up in the field
            if self.siege_cnt >= P['sb_T'] and carried_all >= P['sb_carry'] and still and step - self.last_deliv >= P['sb_gap']:
                self.sb_on = True
            elif carried_all < P['sb_off']:
                self.sb_on = False
        sbreak = self.sb_on and self.siege_cnt >= P['siege_T']
        if sbreak:
            self.sb_cnt += 1
        besiegers = []
        for g_ in list(self.guard):
            if g_ not in workers or workers[g_][1] > 0 or not sbreak or self.guard[g_] not in siege_bases:
                del self.guard[g_]
        if sbreak and R > P['camp_end_from']:
            # home guard: empty collectors that chase the blockers off my doorstep (and never load up)
            besiegers = [ew for ew in enC if ew in still]
            sb_bases = sorted(b for b in siege_bases if any(DIST[workers[ew][0]][b] <= 2 for ew in besiegers))
            for b in sb_bases:
                if b in self.guard.values() or len(self.guard) >= P['sb_guards']:
                    continue
                best = None
                for w in myC:
                    if workers[w][1] > 0 or w in self.guard or w in self.camp:
                        continue
                    d = DIST[workers[w][0]][b]
                    if d <= 6 and (best is None or d < best[0]):
                        best = (d, w)
                if best is not None:
                    self.guard[best[1]] = b
            lack = [b for b in sb_bases if b not in self.guard.values()]
            if lack and len(self.guard) < P['sb_guards']:
                need_sweeper = True
                if nwork >= P['max_units']:
                    # the roster is full: take the slot from a post first, a planter only when trees no longer pay
                    cs = [w for w in self.camp if w in workers and workers[w][1] == 0]
                    if P['sb_recall'] and cs and (keep_planters or not myP or cash < 30):
                        w = min(cs, key=lambda w: DIST[workers[w][0]][lack[0]])
                        del self.camp[w]
                        self.guard[w] = lack[0]
                        need_sweeper = False
                    elif keep_planters and step - self.last_deliv < P['sb_gap_p']:
                        need_sweeper = False
        elif P['siege_spawn'] and not blockading and self.siege_cnt >= P['siege_T'] and carried_all >= P['siege_carry'] and R > 8:
            have = any(workers[w][1] == 0 and min(DIST[workers[w][0]][b] for b in siege_bases) <= 5 for w in myC)
            need_sweeper = not have
            if need_sweeper and nwork >= P['max_units'] and keep_planters and step - self.last_deliv < P['sb_gap_p']:
                need_sweeper = False

        # ---------------- planters ------------------------------------
        L = max(0, min(50, R - 1))
        TV = [-1e9] * N
        if L > 0:
            for q in range(N):
                if q in tree_at or q in base_cells:
                    continue
                tot = 0.0
                for n in NB[q]:
                    if n in tree_at or n in base_cells:
                        continue
                    v = est[n]
                    if v <= 0:
                        continue
                    m = adj_cnt[n]; mine = adj_mine[n]
                    ea = v * (1.0 - (1.0 - ABS * (m + 1)) ** L)
                    if m:
                        eb = v * (1.0 - (1.0 - ABS * m) ** L)
                        tot += ea * (mine + 1) / (m + 1) - eb * mine / m
                        tot += P['steal_w'] * (eb * (m - mine) / m - ea * (m - mine) / (m + 1))
                    else:
                        tot += ea
                TV[q] = tot * P['hold_p'] - P['cell_loss_w'] * est[q]
        SV = {}
        for tid, t in trees.items():
            Lr = max(0, min(50 - t[1], R - 1))
            tot = 0.0
            for n in NB[t[0]]:
                if n in tree_at or n in base_cells:
                    continue
                m = max(1, adj_cnt[n])
                tot += est[n] * (1.0 - (1.0 - ABS * m) ** Lr) / m
            SV[tid] = tot
        plant_tasks = []
        if L > 3:
            for q in range(N):
                tv = TV[q]
                if tv - price >= P['plant_min'] and tv >= P['plant_ratio'] * price:
                    plant_tasks.append((tv - price, q))
            plant_tasks.sort(reverse=True)
            plant_tasks = plant_tasks[:40]
        seize_tasks = []
        for tid, t in trees.items():
            if t[2] != me:
                w = (1.0 + P['steal_w']) * SV[tid] - 20.0
                if w >= P['seize_min']:
                    seize_tasks.append((w, t[0]))
        en_p_pos = [workers[w][0] for w in enP]
        guard_tasks = []
        for q, tid in my_tree_cells.items():
            sv = SV[tid]
            thr = min([DIST[q][p] for p in en_p_pos] or [99])
            if sv >= P['guard_min'] and thr <= 3:
                guard_tasks.append((P['guard_w'] * sv, q))
        spend = 0.0         # cash committed to planting / seizing this turn
        taken = set()
        pd = P['pdisc']
        # process planters closest to a good task first
        porder = sorted(myP, key=lambda w: workers[w][0])
        retirees = set()
        keep_p = P['early_planters'] if self.block_ok else P['retire_keep']
        stock_low = sum(est) < P['retire_stock']
        if need_sweeper and nwork >= P['max_units'] and myP and cash >= 30 and myC:
            # every unit slot is taken and nobody is empty: give up a planter so the base can field a sweeper
            retirees.add(min(myP, key=lambda w: min(DIST[workers[w][0]][workers[cw][0]] for cw in myC)))
            self.no_p_until = step + P['no_p_T']
        elif step >= P['retire_from'] and stock_low and n_en_trees <= P['retire_entrees'] and R >= P['retire_min_left'] and len(myP) > keep_p and cash >= 30 and myC:
            # late trees no longer pay for the unit slot: free it for a collector
            def _pv(w):
                p_ = workers[w][0]
                return max([wv * pd ** DIST[p_][q] for wv, q in plant_tasks] + [wv * pd ** DIST[p_][q] for wv, q in seize_tasks] + [0.0])
            for w in sorted(myP, key=_pv)[:len(myP) - keep_p]:
                retirees.add(w)
        for wid in porder:
            p = workers[wid][0]
            if wid in retirees:
                dests = [p] + NB[p]
                q5 = []
                for a in range(5):
                    z = dests[a]
                    d = min(DIST[z][workers[cw][0]] for cw in myC)
                    q5.append(-float(d) - (1e6 if anyreach[z] else 0.0) - (0.5 if a == 0 else 0.0))
                if p not in tree_at and p not in base_cells and cash - spend >= price:
                    q5[0] = -1e5
                Q[wid] = q5
                continue
            tasks = []
            for wv, q in plant_tasks:
                if q in taken or any(DIST[q][z] <= 1 for z in taken):
                    continue
                tasks.append((wv, q, 0))
            for wv, q in seize_tasks:
                if q not in taken:
                    tasks.append((wv, q, 1))
            for wv, q in guard_tasks:
                if q not in taken:
                    tasks.append((wv, q, 2))
            q5 = [0.0] * 5
            dests = [p] + NB[p]
            best_t = None; best_v = 0.0
            for a in range(5):
                z = dests[a]
                bv = 0.0; bt = None
                for wv, q, kind in tasks:
                    d = DIST[z][q]
                    if a == 0 and d == 0:
                        if kind == 0 and cash - spend < price:
                            v = wv * pd * pd
                        elif kind == 1 and cash - spend < 20:
                            v = wv * pd * pd
                        else:
                            v = wv * pd if kind != 2 else wv * pd
                    elif a == 0:
                        continue
                    else:
                        v = wv * pd ** (d + 2)
                    if v > bv:
                        bv = v; bt = (q, kind)
                q5[a] = bv
                if bv > best_v:
                    best_v = bv; best_t = bt
            # staying on an empty cell with enough cash plants a tree: forbid unless wanted
            if p not in tree_at and p not in base_cells and cash - spend >= price:
                tv = TV[p]
                if not (tv - price >= P['plant_min'] and tv >= P['plant_ratio'] * price):
                    q5[0] = -1e5
            if p in en_tree_cells and cash - spend >= 20 and not any(q == p for _, q in seize_tasks):
                q5[0] = -1e5
            if bel.cls_seen < 63:
                for a in range(1, 5):
                    z = dests[a]
                    q5[a] += P['info_p'] * sum(1 for n in NB[z] if not bel.known[n])
            # safety
            for a in range(5):
                z = dests[a]
                if anyreach[z]:
                    q5[a] -= 1e6
                elif z in cnt2:
                    q5[a] -= 4.0 * len(cnt2[z])
                if z in en_p_stand and a != 0:
                    if z in my_tree_cells:
                        q5[a] += (1.0 + P['steal_w']) * SV[my_tree_cells[z]] - 10.0
                    else:
                        q5[a] -= P['pp_stand']
                elif z in en_p_reach:
                    # head-on planter collisions cost a unit each
                    if not (a == 0 and p in my_tree_cells):
                        q5[a] -= P['pp_cost']
                if z in en_base_set:
                    q5[a] -= 20.0
            if best_t is not None:
                taken.add(best_t[0])
                if q5[0] >= max(q5[1:]) and best_t[0] == p:
                    if best_t[1] == 0 and cash - spend >= price:
                        spend += price
                    elif best_t[1] == 1 and cash - spend >= 20:
                        spend += 20.0
            Q[wid] = q5

        # ---------------- defence / attack bonuses for collectors ---------
        bonus = {wid: [0.0] * 5 for wid in myC}
        new_watch = []
        new_watch2 = []
        cpos = {wid: workers[wid][0] for wid in myC}
        for ewid in enP:
            e = workers[ewid][0]
            on_my_tree = e in my_tree_cells
            key = 'PT' if on_my_tree else 'P'
            ps = self.pstay(key, 4.0, 5.0) if on_my_tree else self.pstay(key, 1.5, 5.0)
            threatened = False
            for wid in myC:
                p = cpos[wid]
                if DIST[p][e] == 1:
                    threatened = True
                    a = NB[p].index(e) + 1
                    val_k = P['kill_p'] + (SV[my_tree_cells[e]] if on_my_tree else 0.0)
                    bonus[wid][a] += ps * val_k
            if threatened or on_my_tree:
                new_watch.append((ewid, e, key))
            elif P['lunge'] > 0:
                near = [wid for wid in myC if DIST[cpos[wid]][e] == 2]
                if near:
                    pa = self.pstay('P2', 0.6, 6.0) * P['lunge']
                    for wid in near:
                        p = cpos[wid]
                        mids = [m for m in NB[p] if DIST[m][e] == 1]
                        for m in mids:
                            bonus[wid][NB[p].index(m) + 1] += pa / len(mids) * P['kill_p']
                    new_watch2.append((ewid, [cpos[wid] for wid in near], 'P2'))
            # intercept: planter next to one of my trees -> step onto / next to the tree
            if not on_my_tree:
                for q in NB[e]:
                    if q in my_tree_cells and SV[my_tree_cells[q]] >= P['guard_min']:
                        best = None
                        for wid in myC:
                            d = DIST[cpos[wid]][q]
                            if d <= 2 and (best is None or d < best[0]):
                                best = (d, wid)
                        if best is not None:
                            wid = best[1]; p = cpos[wid]
                            dests = [p] + NB[p]
                            for a in range(5):
                                if DIST[dests[a]][q] <= 1:
                                    bonus[wid][a] += P['guard_w'] * SV[my_tree_cells[q]]
        for ewid in enC:
            ew = workers[ewid]
            e = ew[0]
            threatened = False
            for wid in myC:
                p = cpos[wid]
                if DIST[p][e] == 1 and workers[wid][1] < ew[1]:
                    threatened = True
                    a = NB[p].index(e) + 1
                    ps = self.pstay('C', 1.0, 5.0)
                    bonus[wid][a] += ps * (ew[1] + P['kill_c'])
            if threatened:
                new_watch.append((ewid, e, 'C'))
            elif P['lunge'] > 0:
                near = [wid for wid in myC if DIST[cpos[wid]][e] == 2 and workers[wid][1] < ew[1]]
                if near:
                    pa = self.pstay('C2', 0.6, 6.0) * P['lunge']
                    for wid in near:
                        p = cpos[wid]
                        mids = [m for m in NB[p] if DIST[m][e] == 1]
                        for m in mids:
                            bonus[wid][NB[p].index(m) + 1] += pa / len(mids) * (ew[1] + P['kill_c'])
                    new_watch2.append((ewid, [cpos[wid] for wid in near], 'C2'))
        # intercept loaded enemy collectors on the last cell before their base
        for ewid in enC:
            ew = workers[ewid]
            if ew[1] <= 0:
                continue
            e = ew[0]
            for b in en_bases:
                if DIST[e][b] != 2:
                    continue
                appr = [a for a in NB[b] if DIST[a][e] == 1]
                seen = []
                pg = self.pstay('G2', 2.0, 5.0)
                for wid in myC:
                    if workers[wid][1] >= ew[1]:
                        continue
                    p = cpos[wid]
                    dests = [p] + NB[p]
                    for a in appr:
                        if a in dests:
                            pg_ = pg
                            if wid in self.camp:
                                # a posted camper loses nothing by lunging unless that uncovers
                                # another loaded enemy waiting at the other entrance it guards
                                exposed = False
                                for a2 in NB[b]:
                                    if a2 == a or DIST[a2][p] > 1:
                                        continue
                                    for ew2id in enC:
                                        if ew2id != ewid and workers[ew2id][1] > workers[wid][1] and DIST[workers[ew2id][0]][a2] == 1:
                                            exposed = True
                                if not exposed:
                                    bonus[wid][dests.index(a)] += P['camp_lunge'] / len(appr)
                                pg_ = max(pg, P['lunge_floor'])
                            bonus[wid][dests.index(a)] += pg_ / len(appr) * (ew[1] + P['kill_c'])
                            if a not in seen:
                                seen.append(a)
                if seen:
                    new_watch2.append((ewid, seen, 'G2'))
        # sweepers: enemy collectors that hunt my loaded collectors or camp at my bases
        if P['base_def'] > 0:
            carried = sum(workers[w][1] for w in myC)
            for ewid in enC:
                ew = workers[ewid]
                e = ew[0]; ce = ew[1]
                tv = 0.0
                if min(DIST[e][b] for b in my_bases) <= 2:
                    tv = P['sweep_base'] + 0.1 * carried
                for wid in myC:
                    w = workers[wid]
                    if w[1] > ce and DIST[w[0]][e] <= 3:
                        tv = max(tv, P['sweep_w'] * w[1] + (P['sweep_base'] if tv > 0 else 0.0))
                if tv <= 0:
                    continue
                tv = min(tv, P['sweep_max'])
                best = None
                for wid in myC:
                    if workers[wid][1] != ce:
                        continue
                    d = DIST[cpos[wid]][e]
                    if d <= 3 and (best is None or d < best[0]):
                        best = (d, wid)
                if best is None:
                    continue
                d, wid = best
                p = cpos[wid]
                z_ = pred_next.get(ewid)
                if z_ is not None and z_ != e:
                    # a mover: meet it on the cell it is about to enter
                    hit = None
                    for wid2 in myC:
                        if workers[wid2][1] == ce and DIST[cpos[wid2]][z_] <= 1 and (hit is None or DIST[cpos[wid2]][z_] < DIST[cpos[hit]][z_]):
                            hit = wid2
                    if hit is not None:
                        p2 = cpos[hit]
                        a_ = 0 if p2 == z_ else NB[p2].index(z_) + 1
                        bonus[hit][a_] += tv + P['tie_cost'] + ce + 25.0
                        continue
                    for a in range(1, 5):
                        if DIST[NB[p][a - 1]][z_] < DIST[p][z_]:
                            bonus[wid][a] += 0.5 * tv + (P['tie_cost'] + ce if DIST[NB[p][a - 1]][e] <= 1 else 0.0)
                    continue
                if d == 1:
                    bonus[wid][NB[p].index(e) + 1] += tv + P['tie_cost'] + ce
                else:
                    for a in range(1, 5):
                        if DIST[NB[p][a - 1]][e] < d:
                            bonus[wid][a] += 0.5 * tv + (P['tie_cost'] + ce if DIST[NB[p][a - 1]][e] <= 1 else 0.0)
        self.watch = new_watch
        self.watch2 = new_watch2
        # hunting: three or more lighter collectors box a heavy enemy collector, then strike.
        # Careful opponents never step on a cell a lighter unit can reach, so a kill needs every
        # one of the target's five cells threatened at once; the strike itself is a small
        # covering game solved for the mixed strategy that maximises the worst-case hit chance.
        # bookkeeping for evaluation: did last turn's strikes kill, which chases are still alive
        for ew_, c_ in self.h_pend.items():
            if ew_ not in workers:
                self.hs['hits'] += 1; self.hs['hit_cargo'] += c_
        self.h_pend = {}
        for ew_ in [x for x, t_ in self.h_act.items() if step - t_ > 2 or x not in workers]:
            del self.h_act[ew_]
        if P['hunt'] and enC and len(myC) >= P['hunt_n'] and not endgame:
            for k_ in [k_ for k_ in self.hunt_t if k_ not in workers]:
                del self.hunt_t[k_]
            used_h = set()
            tg_ = [ew for ew in enC if workers[ew][1] >= P['hunt_min'] and step >= self.hunt_cd.get(ew, 0)
                   and 2 <= min(DIST[workers[ew][0]][b] for b in en_bases) <= P['hunt_gate']]
            tg_.sort(key=lambda ew: -workers[ew][1])
            nt_ = 0
            for ew in tg_:
                if nt_ >= P['hunt_targets']:
                    break
                e = workers[ew][0]; ce = workers[ew][1]
                cells = N1[e]
                lighter = [cpos[w] for w in myC if workers[w][1] < ce]
                safe = [z for z in cells if not any(DIST[q_][z] <= 1 for q_ in lighter)]
                hs = [w for w in myC if w not in used_h and w not in self.guard and workers[w][1] < ce
                      and workers[w][1] <= P['hunt_hmax']
                      and DIST[cpos[w]][e] <= (P['hunt_Rc'] if w in self.camp else P['hunt_R'])
                      and (P['hunt_camp'] or w not in self.camp)]
                if len(hs) < P['hunt_n'] or all(w in self.camp for w in hs):
                    continue
                hs.sort(key=lambda w: (DIST[cpos[w]][e], workers[w][1]))
                hs = hs[:5]
                opts = []; hs2 = []
                for w in hs:
                    pw_ = cpos[w]; ch = workers[w][1]
                    o = []
                    tether = P['hunt_camp'] == 2 and w in self.camp
                    free_c = False
                    if tether and P['hunt_ct'] and not self.orbit_on:
                        # man-marking: with no other cargo around this gate the post may step out a little
                        b_ = min(en_bases, key=lambda b: DIST[b][self.camp[w]])
                        free_c = DIST[e][b_] <= P['hunt_ct_r'] and not (P['hunt_ct_solo'] and any(
                            x != ew and workers[x][1] > P['camp_keep'] and DIST[workers[x][0]][b_] <= P['hunt_ct_r'] for x in enC))
                    for a in range(5):
                        z = pw_ if a == 0 else NB[pw_][a - 1]
                        if tether and a and free_c and DIST[z][self.camp[w]] <= P['hunt_ct_d']:
                            pass
                        elif tether and a and (safe or DIST[z][self.camp[w]] > 1 or self.orbit_on):
                            continue        # a post holder keeps its post; it only lunges when the box is closed
                        if z in en_base_set or (minreach[z] <= ch and not (tether and a == 0)):
                            continue
                        o.append((a, z))
                    if tether and self.orbit_on:
                        o = []              # circling posts are left alone
                    if o:
                        opts.append(o); hs2.append(w)
                if len(hs2) < P['hunt_n'] or all(w in self.camp for w in hs2):
                    continue
                hs = hs2
                plan = None
                if not safe:
                    # boxed: every cell is threatened, the target has to guess
                    pats = {}
                    for combo in product(*[[x for x in o if x[1] in cells] + [None] for o in opts]):
                        dz = [x[1] for x in combo if x is not None]
                        if not dz or len(set(dz)) < len(dz):
                            continue
                        key = tuple(1.0 if z in dz else 0.0 for z in cells)
                        if key not in pats:
                            pats[key] = combo
                    if pats:
                        keys = list(pats)
                        M = np.array(keys, dtype=np.float64)
                        cnt_a = np.zeros(len(keys)); qc = np.ones(5); cov = np.zeros(5)
                        for _ in range(240):
                            a_ = int(np.argmax(M @ (qc / qc.sum())))
                            cnt_a[a_] += 1.0; cov += M[a_]
                            qc[int(np.argmin(cov))] += 1.0
                        pr = cnt_a / cnt_a.sum()
                        r_ = random.random(); acc = 0.0; pick = len(keys) - 1
                        for i_ in range(len(keys)):
                            acc += pr[i_]
                            if r_ <= acc:
                                pick = i_; break
                        combo = pats[keys[pick]]
                        gw_ = min(P['hunt_scap'], P['hunt_sw'] * ce)
                        plan = [(hs[i_], combo[i_][0], gw_) for i_ in range(len(hs)) if combo[i_] is not None
                                and not (hs[i_] in self.camp and combo[i_][0] == 0)]
                        self.hunt_strikes += 1
                        self.hunt_t[ew] = 0
                        self.hs['strikes'] += 1; self.hs['hturns'] += len(plan)
                        self.h_pend[ew] = ce
                        if ew not in self.h_act:
                            self.hs['eps'] += 1
                        if ew not in self.h_boxed:
                            self.h_boxed.add(ew); self.hs['boxed'] += 1
                        self.h_act[ew] = step
                elif P['hunt_tab'] and _hunt_table(P['hunt_tabfile']) is not None:
                    # not boxed yet: plan several turns ahead with the solved boxing game
                    HV, HI, HB = _hunt_table(P['hunt_tabfile'])
                    er, ec = divmod(e, SIZE)

                    def rel_(z, dr_=0, dc_=0):
                        r_, c_ = divmod(z, SIZE)
                        return HI.get((((r_ - er - dr_ + 7) % SIZE) - 7, ((c_ - ec - dc_ + 7) % SIZE) - 7), HB - 1)

                    def val_(i1, i2, i3):
                        a_, b_, c_ = sorted((i1, i2, i3))
                        if c_ == HB - 1 or a_ == b_ or b_ == c_:
                            return 0.0
                        return float(HV[a_ * HB * HB + b_ * HB + c_])
                    best_tri = None
                    nh_ = len(hs)
                    for i1 in range(nh_):
                        for i2 in range(i1 + 1, nh_):
                            for i3 in range(i2 + 1, nh_):
                                if all(hs[x] in self.camp for x in (i1, i2, i3)):
                                    continue
                                v_ = val_(rel_(cpos[hs[i1]]), rel_(cpos[hs[i2]]), rel_(cpos[hs[i3]]))
                                if best_tri is None or v_ > best_tri[0]:
                                    best_tri = (v_, (i1, i2, i3))
                    if best_tri is not None and best_tri[0] >= P['hunt_vmin']:
                        tri = best_tri[1]
                        mvd = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))
                        emoves = [(0 if z == e else NB[e].index(z) + 1) for z in safe]
                        best = None
                        for combo in product(opts[tri[0]], opts[tri[1]], opts[tri[2]]):
                            dz = [x[1] for x in combo]
                            if len(set(dz)) < 3:
                                continue
                            sc = 0.0; lo_ = 9.0
                            for em in emoves:
                                dr_, dc_ = mvd[em]
                                v_ = val_(rel_(dz[0], dr_, dc_), rel_(dz[1], dr_, dc_), rel_(dz[2], dr_, dc_))
                                sc += v_; lo_ = min(lo_, v_)
                            sc = P['hunt_eps'] * sc / len(emoves) + (1.0 - P['hunt_eps']) * lo_
                            if best is None or sc > best[0]:
                                best = (sc, combo)
                        if best is not None:
                            self.hunt_t[ew] = self.hunt_t.get(ew, 0) + 1
                            if self.hunt_t[ew] > P['hunt_pat']:
                                self.hunt_cd[ew] = step + P['hunt_cool']
                                self.hunt_t[ew] = 0
                                self.hs['giveup'] += 1
                            else:
                                gw_ = min(P['hunt_fcap'], P['hunt_fw'] * ce)
                                plan = [(hs[tri[i_]], best[1][i_][0], gw_) for i_ in range(3)
                                        if hs[tri[i_]] not in self.camp or best[1][i_][0] != 0]
                                self.hunt_forms += 1
                                self.hs['form'] += 1; self.hs['hturns'] += len(plan)
                                if ew not in self.h_act:
                                    self.hs['eps'] += 1; self.h_boxed.discard(ew)
                                self.h_act[ew] = step
                else:
                    # not boxed yet: move so that wherever it may go next, it ends up with no safe cell
                    wts = {z: 1.0 for z in safe}
                    h_ = self.ehist.get(ew, ())
                    if e in wts and len(h_) >= 2 and h_[-1] == h_[-2]:
                        wts[e] += 1.0
                    best = None
                    for combo in product(*opts):
                        dz = [x[1] for x in combo]
                        if len(set(dz)) < len(dz):
                            continue
                        sc = 0.0
                        for d_, wt in wts.items():
                            if d_ in dz:
                                continue        # it would not step there (cannot be reached this turn anyway)
                            left = 0
                            for z in N1[d_]:
                                if not any(DIST[x][z] <= 1 for x in dz):
                                    left += 1
                            sc += wt * (1.0 if left == 0 else (0.35 if left == 1 else (0.1 if left == 2 else 0.0)))
                        sc = sc / sum(wts.values()) - 0.001 * sum(DIST[x][e] for x in dz)
                        if best is None or sc > best[0]:
                            best = (sc, combo)
                    if best is not None and best[0] >= (0.0 if any(w in self.rover for w in hs) else P['hunt_fmin']) - 0.03:
                        self.hunt_t[ew] = self.hunt_t.get(ew, 0) + 1
                        if self.hunt_t[ew] > P['hunt_pat']:
                            self.hunt_cd[ew] = step + P['hunt_cool']
                            self.hunt_t[ew] = 0
                            self.hs['giveup'] += 1
                        else:
                            gw_ = min(P['hunt_fcap'], P['hunt_fw'] * ce)
                            plan = [(hs[i_], best[1][i_][0], gw_) for i_ in range(len(hs))
                                    if hs[i_] not in self.camp or best[1][i_][0] != 0]
                            self.hunt_forms += 1
                            self.hs['form'] += 1; self.hs['hturns'] += len(plan)
                            if ew not in self.h_act:
                                self.hs['eps'] += 1; self.h_boxed.discard(ew)
                            self.h_act[ew] = step
                if plan:
                    nt_ += 1
                    for w, a, gw_ in plan:
                        bonus[w][a] += gw_
                        used_h.add(w)
        # campers: empty collectors posted diagonally to enemy bases
        ncamp = P['campers'] if (len(myC) >= (P['camp_from'] if R > P['camp_late'] else P['camp_from_late']) and R > P['camp_until']) else 0
        for wid in list(self.camp):
            if wid not in workers:
                self.camp_lost += 1
                self.camp_loss_t = step
                self.camp_deaths.append(step)
                del self.camp[wid]
                if P['camp_orbit'] == 2 and not self.orbit_on and R > P['camp_end_from'] and \
                        sum(1 for t_ in self.camp_deaths if step - t_ <= P['orbit_win']) >= P['orbit_kill']:
                    # standing posts are being rammed: start circling instead of giving up
                    self.orbit_on = True
                    self.camp_deaths = []
        if self.camp_level is None:
            self.probing = P['probe_n'] > 0
            self.camp_level = P['probe_n'] if self.probing else P['campers']
        if self.probing and self.camp_level > 0:
            # a small probe first: an opponent that rams it with empty units gets no second serving
            if sum(1 for t_ in self.camp_deaths if step - t_ <= P['probe_win']) >= P['probe_kill']:
                self.camp_level = 0
                self.probing = False
        recent = sum(1 for t_ in self.camp_deaths if step - t_ <= P['kill_win'])
        if (recent >= P['kill_n'] or len(self.camp_deaths) >= P['lost_max']) and self.camp_level > 0:
            # the opponent trades units for my posts; give up only if its deliveries still flow
            wnd = P['kill_win']
            got = sum(self.deliv_hist[-wnd:])
            expect = P['eff_rate'] * sum(self.ec_hist[-wnd:])
            if self.flow(wnd) > P['r2_kill']:
                self.camp_level = 0
            elif len(self.camp_deaths) >= P['lost_max']:
                self.camp_deaths = self.camp_deaths[-(P['kill_n'] - 1):]
        if P['siege_hold'] and self.siege_cnt >= P['siege_hold']:
            # they are sitting on my bases: a mutual siege beats a one-sided one, keep mine up
            self.camp_level = P['campers']
            self.probing = False
            self.eval_bad = 0
        if self.camp_lost and step - self.camp_loss_t >= P['camp_decay']:
            self.camp_lost -= 1
            self.camp_loss_t = step
        if self.camp_level is None:
            self.camp_level = P['campers']
        if ncamp:
            ncamp = max(P['camp_min'], min(ncamp, self.camp_level) - self.camp_lost // P['camp_div'])
            # is the blockade actually stopping deliveries?  judge every eval_T turns in position
            inpos = sum(1 for w_, post_ in self.camp.items() if DIST[workers[w_][0]][post_] <= 1)
            if self.probing:
                if self.camp_deaths and step - self.camp_deaths[-1] <= 1:
                    self.probe_on = 0
                elif inpos >= min(ncamp, P['probe_n']):
                    self.probe_on += 1
                    if self.probe_on >= P['probe_T']:
                        self.probing = False
                        self.camp_level = P['campers']
            elif inpos >= max(1, min(3, ncamp)):
                if self.eval_on == 0:
                    self.eval_d0 = self.en_deliv; self.eval_lost0 = self.camp_lost; self.eval_ec = 0.0
                self.eval_on += 1
                self.eval_ec += len(enC)
                if self.eval_on >= P['eval_T']:
                    got = self.en_deliv - self.eval_d0
                    expect = P['eff_rate'] * self.eval_ec
                    ratio = got / max(1.0, expect)
                    ratio = self.flow(P['eval_T'])
                    if ratio > P['r2_hi'] or (ratio > P['r2_lo'] and self.camp_lost - self.eval_lost0 >= 3):
                        self.eval_bad += 1
                        if self.eval_bad >= 2:
                            self.camp_level = max(0, self.camp_level - 2)
                    elif ratio < P['r2_lo']:
                        self.eval_bad = 0
                        self.block_ok = True
                        self.camp_level = P['campers']
                    self.eval_on = 0
        self.ncamp_mid = ncamp
        for wid in list(self.camp):
            if workers[wid][1] > P['camp_keep'] or len(self.camp) > ncamp:
                del self.camp[wid]
        if ncamp:
            posts = []; posts2 = []
            for b in en_bases:
                r_, c_ = divmod(b, SIZE)
                pa = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in ((1, 1), (-1, -1))]
                pb = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in ((1, -1), (-1, 1))]
                # prefer the diagonal pair with less carbon underfoot (campers must stay empty)
                ca = sum(0.0 if x in tree_at else est[x] for x in pa)
                cb = sum(0.0 if x in tree_at else est[x] for x in pb)
                if any(x in self.camp.values() for x in pb) or (ca > cb + 1.0 and not any(x in self.camp.values() for x in pa)):
                    pa, pb = pb, pa
                posts.extend(pa); posts2.extend(pb)
            posts.extend(posts2)
            used = set(self.camp.values())
            for post in posts:
                if len(self.camp) >= ncamp:
                    break
                if post in used:
                    continue
                best = None
                for wid in myC:
                    if wid in self.camp or workers[wid][1] > 0 or wid in self.guard:
                        continue
                    d = DIST[cpos[wid]][post]
                    if wid in self.shadow:
                        # the blockade comes first: a shadow is drafted when nobody else is at hand
                        if P['shadow_pen'] >= 99:
                            continue
                        d += P['shadow_pen']
                    if best is None or d < best[0]:
                        best = (d, wid)
                if best is not None:
                    self.camp[best[1]] = post
                    self.shadow.pop(best[1], None)
                    used.add(post)

        # a post whose partner is missing covers only two of the four entrances: let it take the corner on the
        # side the loaded collectors are coming from (the partner's slot follows, so the pair stays opposite)
        if P['lone'] and self.camp and enC and not self.orbit_on:
            for b_ in en_bases:
                r_, c_ = divmod(b_, SIZE)
                dg_ = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
                opp_ = {dg_[0]: dg_[1], dg_[1]: dg_[0], dg_[2]: dg_[3], dg_[3]: dg_[2]}
                mine_ = [w for w, cell in self.camp.items() if cell in opp_]
                near_ = [w for w in mine_ if DIST[cpos[w]][b_] <= 2]
                if len(near_) != 1:
                    continue
                w0_ = near_[0]
                thr_ = [(workers[ew][0], workers[ew][1]) for ew in enC
                        if workers[ew][1] >= P['mark_min'] and 0 < DIST[workers[ew][0]][b_] <= P['lone_R']]
                if not thr_:
                    continue
                best = None
                for D_ in dg_:
                    sc_ = -P['lone_stick'] * DIST[cpos[w0_]][D_]
                    for E_ in NB[b_]:
                        if DIST[D_][E_] == 1:
                            sc_ += sum(ce_ / (1.0 + DIST[e_][E_]) for e_, ce_ in thr_)
                    if best is None or sc_ > best[0]:
                        best = (sc_, D_)
                if best[1] != self.camp[w0_]:
                    self.camp[w0_] = best[1]
                    self.lone_moves = getattr(self, 'lone_moves', 0) + 1
                for w in mine_:
                    if w != w0_:
                        self.camp[w] = opp_[self.camp[w0_]]
        # marking: a post holder does not wait on a fixed corner; it steps between the base and the loaded
        # collector that is closest to getting in (a careful carrier never walks into a lighter unit's reach)
        self.mark = {}
        if P['mark'] and self.camp and enC:
            thr_ = []
            for ew in enC:
                e_, ce_ = workers[ew][0], workers[ew][1]
                if ce_ < P['mark_min']:
                    continue
                b_ = min(en_bases, key=lambda bb: DIST[e_][bb])
                d_ = DIST[e_][b_]
                if d_ > P['mark_R'] or d_ == 0:
                    continue
                thr_.append((ce_ * (1.0 if d_ <= 2 else (0.6 if d_ == 3 else 0.35)), ew, b_))
            thr_.sort(reverse=True)
            freep_ = set(self.camp)
            for u_, ew, b_ in thr_:
                if len(self.mark) >= P['mark_n'] or not freep_:
                    break
                e_ = workers[ew][0]
                r_, c_ = divmod(b_, SIZE)
                ring_ = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in
                         ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))]
                taken_ = set(v_[1] for v_ in self.mark.values())
                cells_ = [z for z in ring_ if z not in taken_]
                if not cells_:
                    continue
                cell_ = min(cells_, key=lambda z: (DIST[z][e_], DIST[z][b_] == 1))
                best = None
                for w in freep_:
                    if workers[w][1] >= workers[ew][1]:
                        continue
                    dw_ = DIST[cpos[w]][cell_]
                    if dw_ <= P['mark_reach'] and (best is None or dw_ < best[0]):
                        best = (dw_, w)
                if best is not None:
                    self.mark[best[1]] = (ew, cell_)
                    freep_.discard(best[1])
                    self.mark_turns = getattr(self, 'mark_turns', 0) + 1
        # shadows: a planter must stand still for a turn to plant or seize, and a careful one never does so next
        # to a collector.  One empty collector that keeps stepping into its footprints stops it for good.
        for wid in list(self.shadow):
            if wid not in workers or self.shadow[wid] not in workers or workers[wid][1] > P['camp_keep'] or wid in self.camp or wid in self.guard:
                del self.shadow[wid]
        if P['shadow_minfree']:
            while self.shadow and sum(1 for w_ in myC if w_ not in self.camp and w_ not in self.guard and w_ not in self.shadow and w_ not in self.rover) < P['shadow_minfree']:
                # harvesting comes first: release the shadow that is farthest from its planter
                w_ = max(self.shadow, key=lambda x: DIST[cpos[x]][workers[self.shadow[x]][0]])
                del self.shadow[w_]
        n_en_tr_ = sum(1 for t in trees.values() if t[2] != me)
        if n_en_tr_ >= P['shadow_trees']:
            self.sh_seen = True
        want_s = P['shadows'] if (enP and len(myC) >= P['shadow_from'] - (P['shadow_hyst'] if self.shadow else 0) and step >= P['shadow_start'] and R > P['shadow_end']
                                  and self.sh_seen) else 0
        if not want_s:
            self.shadow = {}
        else:
            def sh_pick(ep_, dmax_):
                best_ = None
                if P['shadow_minfree'] and sum(1 for w_ in myC if w_ not in self.camp and w_ not in self.guard and w_ not in self.shadow and w_ not in self.rover) <= P['shadow_minfree']:
                    return None
                for wid in myC:
                    if wid in self.guard or wid in self.rover or wid in self.shadow or workers[wid][1] > 0:
                        continue
                    d = DIST[cpos[wid]][workers[ep_][0]]
                    if wid in self.camp:
                        # a post holder standing next to a planter that just left its base takes over on the spot
                        if d > P['shadow_post_d']:
                            continue
                        d += 0.5
                    if d <= dmax_ and (best_ is None or d < best_[0]):
                        best_ = (d, wid)
                return best_
            # hand the job to a unit that is already on the planter's heels
            if P['shadow_swap']:
                for wid, ep in list(self.shadow.items()):
                    d0_ = DIST[cpos[wid]][workers[ep][0]]
                    if d0_ >= 3:
                        alt_ = sh_pick(ep, min(2.5, d0_ - 2))
                        if alt_ is not None:
                            del self.shadow[wid]
                            self.shadow[alt_[1]] = ep
                            self.camp.pop(alt_[1], None)
            taken_p = set(self.shadow.values())
            for ep in sorted(enP, key=lambda x: min(DIST[workers[x][0]][b] for b in my_bases)):
                if len(self.shadow) >= want_s:
                    break
                if ep in taken_p:
                    continue
                best = sh_pick(ep, 99)
                if best is not None:
                    self.shadow[best[1]] = ep
                    self.camp.pop(best[1], None)
                    taken_p.add(ep)
        # rovers: spare empty collectors that loiter outside the enemy gates and close boxes with the posts
        for wid in list(self.rover):
            if wid not in workers or workers[wid][1] > P['camp_keep'] or wid in self.camp or wid in self.guard:
                del self.rover[wid]
        want_r = P['rovers'] if (ncamp >= 4 and len(self.camp) >= 4 and len(myC) >= P['rover_from'] and not endgame and not self.orbit_on) else 0
        if len(self.rover) > want_r:
            for wid in list(self.rover)[want_r:]:
                del self.rover[wid]
        if want_r:
            for b in en_bases:
                if len(self.rover) >= want_r:
                    break
                if b in self.rover.values() and want_r <= len(en_bases):
                    continue
                best = None
                for wid in myC:
                    if wid in self.camp or wid in self.guard or wid in self.rover or workers[wid][1] > 0:
                        continue
                    d = DIST[cpos[wid]][b]
                    if best is None or d < best[0]:
                        best = (d, wid)
                if best is not None:
                    self.rover[best[1]] = b

        en_occ_c = {workers[ew][0]: workers[ew][1] for ew in enC}
        # ---------------- collectors ----------------------------------
        corder = sorted(myC, key=lambda w: (-workers[w][1], -est[workers[w][0]]))
        run_try = {}

        def post_free(w_):
            return w_ not in self.camp and w_ not in self.guard and w_ not in self.rover and w_ not in self.shadow
        for wid in corder:
            p, c, _, _ = workers[wid]
            if P['displace'] and en_occ:
                occm = claim.copy()
                for z_, c_ in en_occ:
                    if c_ <= c:
                        occm[z_] *= 0.5
                G = Gfree * occm[None, None, :]
            else:
                G = Gfree * claim[None, None, :]
            taxc = np.array(tax_me, dtype=np.float64) * (0.1 * max(c, 5.0))
            Cr = taxc.copy()
            C1 = taxc.copy()
            C0 = taxc - P['info_c'] * info
            if enC and c > 0:
                a1 = P['avoid1'] * (c + 30.0); a2 = P['avoid2'] * (c + 30.0)
                for z in range(N):
                    if minreach[z] <= c and z not in my_base_set:
                        C1[z] += a1
                        Cr[z] += a2
            # soft risk for the next two arrivals
            if enC:
                for z, lst in cnt2.items():
                    n2 = sum(1 for x in lst if x <= c)
                    if n2:
                        pen = P['soft_w'] * (c + 30.0) * (1.0 if n2 >= 3 else (0.25 if n2 == 2 else 0.06))
                        C0[z] += pen
                        C1[z] += 0.5 * pen
            if endgame:
                lam = 0.0
                term = np.full((S1, N), -1e4, dtype=np.float64)
                term[:, mb] = 0.0
                base_val = np.full(len(my_bases), float(c))
                base_val = base_val[None, :].repeat(S1, axis=0)
            else:
                lam = rho * c
                term = np.empty((S1, N), dtype=np.float64)
                term[:] = -lam * dbase * P['term_k']
                if c > 0:
                    base_val = (Wfree[mb] + mu * c)[None, :].repeat(S1, axis=0)
                else:
                    base_val = None
            V1, pol = run_dp(G, lam, C0, C1, Cr, base_val, term, True)
            if Hh >= 2:
                arr = gamma * V1[0] - C0
                qs = G[0][0][p] + gamma * V1[1][p] - C0[p] - lam
            else:
                arr = gamma * term[0] - C0
                qs = G[0][0][p] + gamma * term[0][p] - C0[p]
            dests = [p] + NB[p]
            q5 = [float(qs)] + [float(arr[z]) - lam for z in NB[p]]
            if endgame and c > 0:
                # delivered cargo counts fully when stepping into a base
                for a in range(1, 5):
                    if dests[a] in my_base_set:
                        q5[a] = float(c)
                if p in my_base_set:
                    q5[0] = float(c)
            if P['gate'] and gate_out and not endgame and (not blockading or (sbreak and P['sb_gate'])):
                hb = min(my_bases, key=lambda b: DIST[p][b])
                if hb in gate_out:
                    if c >= P['gate_min']:
                        # besieged: wait on the cell straight out from an entrance, step in when it is uncovered
                        gw = min(P['gate_cap'], P['gate_w'] * c)
                        outs = gate_out[hb]
                        here = [j_ for x_, j_ in outs if x_ == p]
                        if DIST[p][hb] == 1:
                            pass
                        elif here:
                            q5[NB[p].index(here[0]) + 1] += gw
                            q5[0] += 0.6 * gw
                            for a in range(1, 5):
                                if dests[a] != here[0]:
                                    q5[a] += 0.3 * gw
                        else:
                            dnow = min(DIST[p][x_] for x_, j_ in outs)
                            for a in range(5):
                                dn = min(DIST[dests[a]][x_] for x_, j_ in outs)
                                q5[a] += 0.4 * gw * (dnow - dn)
                    elif c == 0 and DIST[p][hb] <= 3 and est[p] > 1.0 and p not in tree_at:
                        q5[0] -= 20.0       # a sweeper that loads up stops being one
            runme = bool(run_cells) and c > 0 and post_free(wid) and (wid == runner or (run_all and c >= P['gate_min']))
            if runme:
                hb_ = min(siege_bases, key=lambda b: DIST[p][b])
                if run_all and wid != runner and DIST[p][hb_] > 5:
                    runme = False
                else:
                    gw_ = min(P['gate_cap'], max(40.0, P['gate_w'] * c))
                    dn_ = DIST[p][hb_]
                    if P['run_mode'] < 2 or wid == runner:
                        for a in range(5):
                            q5[a] += gw_ * (dn_ - DIST[dests[a]][hb_])
            bn = bonus[wid]
            post = self.camp.get(wid)
            tie_cost = P['tie_cost']
            gb = self.guard.get(wid)
            if gb is not None:
                tie_cost = P['camp_tie']
                tgc = None; bd_ = 99
                for ew in besiegers:
                    e_ = workers[ew][0]
                    if DIST[e_][gb] > 3:
                        continue
                    z_ = pred_next.get(ew, e_)
                    d_ = DIST[p][z_]
                    if d_ < bd_:
                        bd_ = d_; tgc = z_
                for a in range(5):
                    z = dests[a]
                    if tgc is not None:
                        q5[a] = -P['camp_w'] * DIST[z][tgc]
                    else:
                        q5[a] = -P['camp_w'] * abs(DIST[z][gb] - 1)
                    if a == 0 and est[p] > 1.0 and p not in tree_at and p not in my_base_set:
                        q5[a] -= 60.0
                    if z in my_base_set:
                        q5[a] -= 10.0
            sp_ = self.shadow.get(wid)
            if sp_ is not None:
                tie_cost = P['orbit_tie']
                e_ = workers[sp_][0]
                self.sh_turns += 1
                self.sh_adj += DIST[p][e_] <= 1
                for a in range(5):
                    z = dests[a]
                    q5[a] = -P['camp_w'] * DIST[z][e_] - (40.0 if a == 0 and est[p] > 1.0 and p not in tree_at else 0.0)
            rb_ = self.rover.get(wid)
            if rb_ is not None:
                tie_cost = P['orbit_tie']
                for a in range(5):
                    z = dests[a]
                    q5[a] = -P['camp_w'] * abs(DIST[z][rb_] - P['rover_d']) - (40.0 if a == 0 and est[p] > 1.0 and p not in tree_at else 0.0)
            if post is not None:
                tie_cost = P['camp_tie']
                tgt = post
                if P['camp_orbit'] == 1 or (P['camp_orbit'] == 2 and self.orbit_on):
                    # circle the enemy base instead of standing still (never collects, hard to ram)
                    b_ = min(en_bases, key=lambda b: DIST[b][post])
                    r_, c_ = divmod(b_, SIZE)
                    ring = [((r_ + dr) % SIZE) * SIZE + (c_ + dc) % SIZE for dr, dc in
                            ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))]
                    if post in ring:
                        tgt = ring[(step + 1 + ring.index(post)) % 8]
                        tie_cost = P['orbit_tie']
                mk_ = self.mark.get(wid)
                if mk_ is not None:
                    tgt = mk_[1]
                    tie_cost = P['orbit_tie']
                    e_ = workers[mk_[0]][0]
                    for a in range(5):
                        z = dests[a]
                        q5[a] = -P['camp_w'] * DIST[z][tgt] - (2.0 if a == 0 and est[p] > 1 and p != tgt else 0.0)
                        if P['mark_chase'] and z == e_ and e_ not in en_base_set:
                            q5[a] += P['mark_chase']
                else:
                    for a in range(5):
                        q5[a] = -P['camp_w'] * DIST[dests[a]][tgt] - (2.0 if a == 0 and est[p] > 1 and p != post else 0.0)
            for a in range(5):
                z = dests[a]
                q5[a] += bn[a]
                m = minreach[z]
                if z in en_base_set:
                    q5[a] -= 1e7
                elif m < c:
                    if z in my_base_set:
                        q5[a] -= 15.0
                    else:
                        if endgame and R <= 6:
                            q5[a] -= (0.7 if a == 0 else 0.45) * (c + 30.0)
                        elif a and runme and z in run_cells and \
                                all(workers[ew][1] >= c or workers[ew][0] in still_pos for ew in enC if DIST[workers[ew][0]][z] <= 1):
                            if P['run_mode'] == 3 and wid != runner:
                                # expected loss at the strike rate seen for this cargo class (never below a lighter class)
                                pr_ = 0.0
                                for b_, (a0_, b0_) in enumerate(((1.0, 1.0), (1.5, 0.5), (2.0, 0.5))):
                                    pr_ = max(pr_, (self.run_b[b_][1] + a0_) / (self.run_b[b_][0] + self.run_b[b_][1] + a0_ + b0_))
                                    if c < (P['run_b1'], P['run_b2'], 1e18)[b_]:
                                        break
                                q5[a] -= P['run_ev'] * pr_ * (2.0 * c + 30.0)
                            elif P['run_mode'] == 2 and wid != runner:
                                # expected loss: the cargo goes to the opponent (twice its value) plus the unit
                                q5[a] -= P['run_ev'] * (self.run_fail + 1.0) / (self.run_ok + self.run_fail + 2.0) * (2.0 * c + 30.0)
                            else:
                                q5[a] -= (P['run_k1'] if (run_all and wid != runner) else P['run_k0']) * (c + 30.0)
                            run_try[wid] = a
                        else:
                            # lethal: ordered by how likely the hunter really goes there
                            q5[a] -= 1e6 + (500.0 if a == 0 else 0.0)
                elif m == c:
                    q5[a] -= tie_cost + c
                    if a and post is not None and P['camp_occ'] and z in en_occ_c and en_occ_c[z] == c:
                        # an equal unit is standing on that cell right now: walking into it is a near-certain even trade far from home
                        q5[a] -= P['camp_occ']
            Q[wid] = q5
            # claim the cells this collector plans to harvest
            a0 = max(range(5), key=lambda a: q5[a])
            z = dests[a0]; s = 0
            if a0 == 0:
                claim[z] *= 0.7; s = 1
            for t in range(1, Hh):
                stay_m, dirs = pol[t - 1]
                if z in my_base_set and c > 0:
                    break
                if stay_m[min(s, S1 - 1)][z]:
                    claim[z] *= 0.7
                    s += 1
                else:
                    z = NB[z][int(dirs[z])]
                    s = 0

        # ---------------- joint resolution ----------------------------
        units = list(Q.keys())
        pos = {wid: workers[wid][0] for wid in units}
        choice = {}
        for wid in units:
            q5 = Q[wid]
            choice[wid] = max(range(5), key=lambda a: q5[a])

        def dest_of(wid):
            a = choice[wid]
            return pos[wid] if a == 0 else NB[pos[wid]][a - 1]

        for _ in range(40):
            dm = {}
            for wid in units:
                dm.setdefault(dest_of(wid), []).append(wid)
            conflict = [(z, lst) for z, lst in dm.items() if len(lst) > 1]
            if not conflict:
                break
            for z, lst in conflict:
                if endgame and R <= 1 and z in my_base_set:
                    continue        # last move: cargo is banked even if own units collide
                # the unit that loses most by yielding keeps the cell
                regs = []
                for wid in lst:
                    q5 = Q[wid]
                    a = choice[wid]
                    alt = max([q5[b] for b in range(5) if b != a])
                    regs.append((q5[a] - alt, wid))
                regs.sort(reverse=True)
                for _, wid in regs[1:]:
                    Q[wid][choice[wid]] = -1e8
                    q5 = Q[wid]
                    choice[wid] = max(range(5), key=lambda a: q5[a])
        if retirees:
            cdest = {}
            for cw in myC:
                cdest.setdefault(dest_of(cw), []).append(cw)
            for wid in retirees:
                p = pos[wid]
                dests = [p] + NB[p]
                for a in range(5):
                    z = dests[a]
                    if len(cdest.get(z, ())) == 1 and z not in en_base_set:
                        choice[wid] = a
                        break
        actions = {}
        self.last_dest = {}
        final_dest = set()
        for wid in units:
            a = choice[wid]
            z = dest_of(wid)
            final_dest.add(z)
            self.last_dest[wid] = z
            if run_try.get(wid) == a and a:
                self.run_pend[wid] = workers[wid][1]
                self.run_last = step
                self.run_cnt += 1
            if a:
                actions[wid] = ACT[a - 1]

        # ---------------- recruiting ----------------------------------
        budget = cash - spend
        nC = len(myC); nP = len(myP)
        free_bases = [(bid, b) for bid, b in bases[me] if b not in final_dest]
        if step == 0:
            order = P['open']
        else:
            order = None
        k = 0
        if need_sweeper:
            for bid, b in sorted(free_bases, key=lambda t_: 0 if t_[1] in siege_bases else 1):
                if nwork < P['max_units'] and budget >= 30:
                    actions[bid] = 'RECCOLLECTOR'
                    budget -= 30; nwork += 1; nC += 1
                    break
        for bid, b in free_bases:
            if bid in actions:
                continue
            late_post = False
            if R <= P['recruit_until']:
                # late in the game a new unit only pays as a post holder: it keeps the opponent's hoard outside
                late_post = bool(P['recruit_post']) and R > P['recruit_post'] and self.ncamp_mid > len(self.camp) and budget >= 150
                if not late_post:
                    break
            if nwork >= P['max_units'] or budget < 30:
                break
            if order is not None:
                typ = order[k] if k < len(order) else None
                k += 1
                if typ is None:
                    break
            else:
                want_p = 0 if step < self.no_p_until else (P['early_planters'] if (step < P['retire_from'] - 12 or self.block_ok or not stock_low or n_en_trees > P['retire_entrees']) else P['retire_keep'])
                if P['p_extra'] and step >= self.no_p_until and n_en_trees >= P['p_extra_trees'] and R > 60:
                    want_p = max(want_p, P['early_planters'] + P['p_extra'])
                if nP < want_p and nC >= P['planter_from'] * (nP + 0) and nC >= 2 and not late_post:
                    typ = 'P'
                else:
                    typ = 'C'
            # never spawn into an enemy collector stepping on the base with equal cargo
            actions[bid] = 'RECPLANTER' if typ == 'P' else 'RECCOLLECTOR'
            budget -= 30
            nwork += 1
            if typ == 'P':
                nP += 1
            else:
                nC += 1
        return actions


_BOT = Bot()


def agent(observation, configuration):
    global _BOT
    if observation['step'] == 0:
        _BOT = Bot()
    try:
        return _BOT.act(observation, configuration)
    except Exception:
        if os.environ.get('CARBON_DEBUG'):
            raise
        return {}
