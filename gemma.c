/* Gemma 1, 2 and 3 implemented in a single file of pure C.
 *
 * This is a self-contained inference engine for Google's Gemma family of
 * language models. It supports Gemma 1, 2 and 3 across all their variants,
 * including both the base text models and the multimodal vision-language
 * models that incorporate a SigLIP vision tower.
 *
 * The entire implementation is contained in this one file with no external
 * dependencies beyond the C standard library and OpenMP for parallelization
 * of matrix operations (which is optional). It's extremely easy to compile
 * on any system with a C compiler and optional OpenMP support. The code is
 * written with a strong emphasis on performance, utilizing hand-tuned matrix
 * multiplication kernels and efficient memory management patterns.
 *
 * Features include:
 *
 *   W8A8 quantization.
 *
 *   Hybrid attention mechanisms combining both sliding-window attention
 *   for local context and full attention for global context, following
 *   the approach used in Gemma 2 and 3.
 *
 *   An optional SigLIP vision encoder for multimodal models.
 *
 *   A lot of sampling strategies.
 *
 *   Support for float32 & float16 & bfloat16.
 *
 *   A minimal BPE tokenizer.
 *
 * The model files are expected to be in a custom binary format generated
 * by the accompanying `export.py`. The format stores all model weights,
 * configuration parameters, tokenizer vocabulary, and BPE merge rules in
 * a single file. The export script handles the conversion from the original
 * PyTorch checkpoint format into this inference-ready format, applying
 * quantization and reordering weights for optimal performance.
 *
 * To compile the code, simply run:
 *
 *   gcc -Ofast -march=native -mtune=native -fopenmp gemma.c -lm -o gemma
 *
 * Basic usage of the compiled binary is straightforward:
 *
 *   ./gemma model.bin -i "Hello I'm a language model, "
 *
 * This loads the model file, processes the prompt, and completes the prompt.
 * The model automatically handles all tokenization and decoding internally.
 *
 * For interactive multi-turn conversations, use the chat mode:
 *
 *   ./gemma model.bin -c
 *
 * In chat mode, the program maintains conversation history across turns,
 * following the Gemma prompt format with <start_of_turn> and <end_of_turn>
 * markers. Each user turn is prefixed with "User: " and the model responds
 * with "Model: ".
 *
 * Multimodal models can process images by specifying the image path in the
 * prompt using the @image{...} syntax. e.g.:
 *
 *   ./gemma model.bin -i "Looking at @image{photo.jpg}, we can see"
 *
 * or you can also use it in chat mode:
 *
 *   User: Describe what you see in @image{photo.jpg}.
 *
 * The image is loaded, resized to the model's expected input dimensions,
 * processed through the vision encoder, and the resulting soft tokens are
 * injected into the text token sequence.
 *
 * You can also disable the multimodal part using the `-d` / `--disable-mm` to
 * argument reduce some memory usage.
 *
 * Most of the standard inference controls are available through command-line
 * options: sequence length, temperature, top-k and top-p sampling, repetition
 * penalty, and random seed.
 *
 * The code is designed to be memory-efficient, reusing buffers across
 * inference steps to minimize allocation overhead. The KV cache is
 * preallocated based on the maximum sequence length, and all intermediate
 * tensors are stored in reusable buffers.
 *
 * For developers looking to understand or extend the code, the
 * implementation is organized into clear functional sections: model loading,
 * tokenization, attention computation, feedforward networks, sampling
 * strategies, and the main inference loop. The matrix multiplication kernels
 * are isolated and can be replaced / optimized independently.
 *
 * TerryGuo 09/05/2026
 */

#include <assert.h>
#include <emmintrin.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This giant blob is a self-contained, pre-processed implementation of
 * "stb_image" and "stb_image_resize2". :D
 *
 * I want to keep this project in a single .c file, to avoid requiring users to
 * obtain the original stb headers, and to keep the source code free of tens of
 * thousands of lines of third-party code, I just took the source code of the
 * two libraries, compressed / minified them into a "code blob" and embedded
 * directly here. The functions declared `extern` remain visible. (This blob
 * polluted the global namespace quite seriously, but I'm good with that ;-)
 *
 * The blob is treated as a black box, please do not edit it.
 */

// NOLINTBEGIN        // Tell clang-tidy to shutup
// Tell clang-format to ignore this blob
// clang-format off

typedef unsigned char l;typedef struct{int(*l)(void*user,char*data,int size);//
void(*u)(void*user,int n);int(*ab)(void*user);}u;typedef struct{uint32_t l,u;//
int ab,ac;u ad;void*ae;int ag;int ah;l ai[128];int aj;l*ak,*al;l*am,*an;}ab;///
static void ac(ab*tn);static void ad(ab*tn,u*to,void*tp){tn->ad=*to;tn->ae=tp;
tn->ah=sizeof(tn->ai);tn->ag=1;tn->aj=0;tn->ak=tn->am=tn->ai;ac(tn);tn->an=tn->
al;}static int ae(void*tn,char*to,int tp){return(int)fread(to,1,tp,(FILE*)tn);}
static void ag(void*tn,int to){int tp;fseek((FILE*)tn,to,SEEK_CUR);tp=fgetc((//
FILE*)tn);if(tp!=EOF){ungetc(tp,(FILE*)tn);}}static int ah(void*tn){return feof
((FILE*)tn)||ferror((FILE*)tn);}static u ai={ae,ag,ah,};static void aj(ab*tn,//
FILE*to){ad(tn,&ai,(void*)to);}static void ak(ab*tn){tn->ak=tn->am;tn->al=tn->
an;}enum{STBI_ORDER_RGB,STBI_ORDER_BGR};typedef struct{int l;int u;int ab;}al;
static int am(ab*tn);static void*an(ab*tn,int*to,int*tp,int*tq,int tr,al*ts);//
static int ao(ab*tn);static void*ap(ab*tn,int*to,int*tp,int*tq,int tr,al*ts);//
static _Thread_local const char*aq;static int ar(const char*tn){aq=tn;return 0;
}static void*as(size_t tn){return malloc(tn);}static int at(int tn,int to){if(
to<0)return 0;return tn<=INT_MAX-to;}static int au(int tn,int to){if(tn<0||to<0
)return 0;if(to==0)return 1;return tn<=INT_MAX/to;}static int av(int tn,int to,
int tp){return au(tn,to)&&at(tn*to,tp);}static int aw(int tn,int to,int tp,int
tq){return au(tn,to)&&au(tn*to,tp)&&at(tn*to*tp,tq);}static void*ax(int tn,int
to,int tp){if(!av(tn,to,tp))return NULL;return as(tn*to+tp);}static void*ay(int
tn,int to,int tp,int tq){if(!aw(tn,to,tp,tq))return NULL;return as(tn*to*tp+tq)
;}static int az(int tn,int to){if((tn>=0)!=(to>=0))return 1;if(tn<0&&to<0)/////
return tn>=INT_MIN-to;return tn<=INT_MAX-to;}static int ba(int tn,int to){if(to
==0||to==-1)return 1;if((tn>=0)==(to>=0))return tn<=SHRT_MAX/to;if(to<0)return
tn<=SHRT_MIN/to;return tn>=SHRT_MIN/to;}extern void stbi_image_free(void*tn){//
free(tn);}static int bb=0;static _Thread_local int bc,bd;static void*be(ab*tn,
int*to,int*tp,int*tq,int tr,al*ts,int tt){memset(ts,0,sizeof(*ts));ts->l=8;ts->
ab=STBI_ORDER_RGB;ts->u=0;if(ao(tn))return ap(tn,to,tp,tq,tr,ts);(void)sizeof(
tt);if(am(tn))return an(tn,to,tp,tq,tr,ts);return((unsigned char*)(size_t)(ar(
"unknown image type")?NULL:NULL));}static l*bg(uint16_t*tn,int to,int tp,int tq
){int tr;int ts=to*tp*tq;l*tt;tt=(l*)as(ts);if(tt==NULL)return((unsigned char*)
(size_t)(ar("outofmem")?NULL:NULL));for(tr=0;tr<ts;++tr)tt[tr]=(l)((tn[tr]>>8)&
0xFF);free(tn);return tt;}static void bh(void*tn,int to,int tp,int tq){int tr;
size_t ts=(size_t)to*tq;l tt[2048];l*tu=(l*)tn;for(tr=0;tr<(tp>>1);tr++){l*tv=
tu+tr*ts;l*tw=tu+(tp-tr-1)*ts;size_t tx=ts;while(tx){size_t ty=(tx<sizeof(tt))?
tx:sizeof(tt);memcpy(tt,tv,ty);memcpy(tv,tw,ty);memcpy(tw,tt,ty);tv+=ty;tw+=ty;
tx-=ty;}}}static unsigned char*bi(ab*tn,int*to,int*tp,int*tq,int tr){al ts;void
*tt=be(tn,to,tp,tq,tr,&ts,8);if(tt==NULL)return NULL;assert(ts.l==8||ts.l==16);
if(ts.l!=8){tt=bg((uint16_t*)tt,*to,*tp,tr==0?*tq:tr);ts.l=8;}if((bd?bc:bb)){//
int tu=tr?tr:*tq;bh(tt,*to,*tp,tu*sizeof(l));}return(unsigned char*)tt;}static
FILE*bj(char const*tn,char const*to){FILE*tp;tp=fopen(tn,to);return tp;}extern
l*stbi_load_from_file(FILE*tn,int*to,int*tp,int*tq,int tr){unsigned char*ts;ab
tt;aj(&tt,tn);ts=bi(&tt,to,tp,tq,tr);if(ts){fseek(tn,-(int)(tt.al-tt.ak),//////
SEEK_CUR);}return ts;}extern l*stbi_load(char const*tn,int*to,int*tp,int*tq,int
tr){FILE*ts=bj(tn,"rb");unsigned char*tt;if(!ts)return((unsigned char*)(size_t)
(ar("can't fopen")?NULL:NULL));tt=stbi_load_from_file(ts,to,tp,tq,tr);fclose(ts
);return tt;}enum{STBI__SCAN_load=0,STBI__SCAN_type,STBI__SCAN_header};static//
void ac(ab*to){int tp=(to->ad.l)(to->ae,(char*)to->ai,to->ah);to->aj+=(int)(to
->ak-to->am);if(tp==0){to->ag=0;to->ak=to->ai;to->al=to->ai+1;*to->ak=0;}else{
to->ak=to->ai;to->al=to->ai+tp;}}static l bk(ab*tn){if(tn->ak<tn->al)return*tn
->ak++;if(tn->ag){ac(tn);return*tn->ak++;}return 0;}static int bl(ab*tn){if(tn
->ad.l){if(!(tn->ad.ab)(tn->ae))return 0;if(tn->ag==0)return 1;}return tn->ak>=
tn->al;}static void bm(ab*tn,int to){if(to==0)return;if(to<0){tn->ak=tn->al;///
return;}if(tn->ad.l){int tp=(int)(tn->al-tn->ak);if(tp<to){tn->ak=tn->al;(tn->
ad.u)(tn->ae,to-tp);return;}}tn->ak+=to;}static int bn(ab*tn,l*to,int tp){if(tn
->ad.l){int tq=(int)(tn->al-tn->ak);if(tq<tp){int tr,ts;memcpy(to,tn->ak,tq);ts
=(tn->ad.l)(tn->ae,(char*)to+tq,tp-tq);tr=(ts==(tp-tq));tn->ak=tn->al;return tr
;}}if(tn->ak+tp<=tn->al){memcpy(to,tn->ak,tp);tn->ak+=tp;return 1;}else return
0;}static int bo(ab*tn){int to=bk(tn);return(to<<8)+bk(tn);}static uint32_t bp(
ab*tn){uint32_t to=bo(tn);return(to<<16)+bo(tn);}static l bq(int tn,int to,int
tp){return(l)(((tn*77)+(to*150)+(29*tp))>>8);}static unsigned char*br(unsigned
char*tn,int to,int tp,unsigned int tq,unsigned int tr){int ts,tt;unsigned char*
tu;if(tp==to)return tn;assert(tp>=1&&tp<=4);tu=(unsigned char*)ay(tp,tq,tr,0);
if(tu==NULL){free(tn);return((unsigned char*)(size_t)(ar("outofmem")?NULL:NULL)
);}for(tt=0;tt<(int)tr;++tt){unsigned char*tv=tn+tt*tq*to;unsigned char*tw=tu+
tt*tq*tp;switch(((to)*8+(tp))){case((1)*8+(2)):for(ts=tq-1;ts>=0;--ts,tv+=1,tw
+=2){tw[0]=tv[0];tw[1]=255;}break;case((1)*8+(3)):for(ts=tq-1;ts>=0;--ts,tv+=1,
tw+=3){tw[0]=tw[1]=tw[2]=tv[0];}break;case((1)*8+(4)):for(ts=tq-1;ts>=0;--ts,tv
+=1,tw+=4){tw[0]=tw[1]=tw[2]=tv[0];tw[3]=255;}break;case((2)*8+(1)):for(ts=tq-1
;ts>=0;--ts,tv+=2,tw+=1){tw[0]=tv[0];}break;case((2)*8+(3)):for(ts=tq-1;ts>=0;
--ts,tv+=2,tw+=3){tw[0]=tw[1]=tw[2]=tv[0];}break;case((2)*8+(4)):for(ts=tq-1;ts
>=0;--ts,tv+=2,tw+=4){tw[0]=tw[1]=tw[2]=tv[0];tw[3]=tv[1];}break;case((3)*8+(4)
):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=4){tw[0]=tv[0];tw[1]=tv[1];tw[2]=tv[2];tw[3]
=255;}break;case((3)*8+(1)):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=1){tw[0]=bq(tv[0],
tv[1],tv[2]);}break;case((3)*8+(2)):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=2){tw[0]=
bq(tv[0],tv[1],tv[2]);tw[1]=255;}break;case((4)*8+(1)):for(ts=tq-1;ts>=0;--ts,
tv+=4,tw+=1){tw[0]=bq(tv[0],tv[1],tv[2]);}break;case((4)*8+(2)):for(ts=tq-1;ts
>=0;--ts,tv+=4,tw+=2){tw[0]=bq(tv[0],tv[1],tv[2]);tw[1]=tv[3];}break;case((4)*8
+(3)):for(ts=tq-1;ts>=0;--ts,tv+=4,tw+=3){tw[0]=tv[0];tw[1]=tv[1];tw[2]=tv[2];}
break;default:assert(0);free(tn);free(tu);return((unsigned char*)(size_t)(ar(//
"unsupported")?NULL:NULL));}}free(tn);return tu;}static uint16_t bs(int tn,int
to,int tp){return(uint16_t)(((tn*77)+(to*150)+(29*tp))>>8);}static uint16_t*bt(
uint16_t*tn,int to,int tp,unsigned int tq,unsigned int tr){int ts,tt;uint16_t*
tu;if(tp==to)return tn;assert(tp>=1&&tp<=4);tu=(uint16_t*)as(tp*tq*tr*2);if(tu
==NULL){free(tn);return(uint16_t*)((unsigned char*)(size_t)(ar("outofmem")?NULL
:NULL));}for(tt=0;tt<(int)tr;++tt){uint16_t*tv=tn+tt*tq*to;uint16_t*tw=tu+tt*tq
*tp;switch(((to)*8+(tp))){case((1)*8+(2)):for(ts=tq-1;ts>=0;--ts,tv+=1,tw+=2){
tw[0]=tv[0];tw[1]=0xffff;}break;case((1)*8+(3)):for(ts=tq-1;ts>=0;--ts,tv+=1,tw
+=3){tw[0]=tw[1]=tw[2]=tv[0];}break;case((1)*8+(4)):for(ts=tq-1;ts>=0;--ts,tv+=
1,tw+=4){tw[0]=tw[1]=tw[2]=tv[0];tw[3]=0xffff;}break;case((2)*8+(1)):for(ts=tq-
1;ts>=0;--ts,tv+=2,tw+=1){tw[0]=tv[0];}break;case((2)*8+(3)):for(ts=tq-1;ts>=0;
--ts,tv+=2,tw+=3){tw[0]=tw[1]=tw[2]=tv[0];}break;case((2)*8+(4)):for(ts=tq-1;ts
>=0;--ts,tv+=2,tw+=4){tw[0]=tw[1]=tw[2]=tv[0];tw[3]=tv[1];}break;case((3)*8+(4)
):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=4){tw[0]=tv[0];tw[1]=tv[1];tw[2]=tv[2];tw[3]
=0xffff;}break;case((3)*8+(1)):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=1){tw[0]=bs(tv[
0],tv[1],tv[2]);}break;case((3)*8+(2)):for(ts=tq-1;ts>=0;--ts,tv+=3,tw+=2){tw[0
]=bs(tv[0],tv[1],tv[2]);tw[1]=0xffff;}break;case((4)*8+(1)):for(ts=tq-1;ts>=0;
--ts,tv+=4,tw+=1){tw[0]=bs(tv[0],tv[1],tv[2]);}break;case((4)*8+(2)):for(ts=tq-
1;ts>=0;--ts,tv+=4,tw+=2){tw[0]=bs(tv[0],tv[1],tv[2]);tw[1]=tv[3];}break;case((
4)*8+(3)):for(ts=tq-1;ts>=0;--ts,tv+=4,tw+=3){tw[0]=tv[0];tw[1]=tv[1];tw[2]=tv[
2];}break;default:assert(0);free(tn);free(tu);return(uint16_t*)((unsigned char*
)(size_t)(ar("unsupported")?NULL:NULL));}}free(tn);return tu;}typedef struct{l
l[1<<9];uint16_t u[256];l ab[256];l ac[257];unsigned int ad[18];int ae[17];}bu;
typedef struct{ab*l;bu u[4];bu ab[4];uint16_t ac[4][64];int16_t ad[4][1<<9];int
ae,ag;int ah,ai;int aj,ak;struct{int l;int u,ab;int ac;int ad,ae;int ag;int ah,
ai,aj,ak;l*al;void*am,*an;l*ao;short*ap;int aq,ar;}al[4];uint32_t am;int an;///
unsigned char ao;int ap;int aq;int ar;int as;int at;int au;int av;int aw;int ax
;int ay;int az,ba[4];int bb,bc;void(*bd)(l*out,int out_stride,short data[64]);
void(*be)(l*out,const l*y,const l*pcb,const l*pcr,int count,int step);l*(*bg)(l
*out,l*in_near,l*in_far,int w,int hs);}bv;static int bx(bu*tn,int*to){int tp,tq
,tr=0;unsigned int ts;for(tp=0;tp<16;++tp){for(tq=0;tq<to[tp];++tq){tn->ac[tr++
]=(l)(tp+1);if(tr>=257)return ar("bad size list");}}tn->ac[tr]=0;ts=0;tr=0;for(
tq=1;tq<=16;++tq){tn->ae[tq]=tr-ts;if(tn->ac[tr]==tq){while(tn->ac[tr]==tq)tn->
u[tr++]=(uint16_t)(ts++);if(ts-1>=(1u<<tq))return ar("bad code lengths");}tn->
ad[tq]=ts<<(16-tq);ts<<=1;}tn->ad[tq]=0xffffffff;memset(tn->l,255,1<<9);for(tp=
0;tp<tr;++tp){int tt=tn->ac[tp];if(tt<=9){int tu=tn->u[tp]<<(9-tt);int tv=1<<(9
-tt);for(tq=0;tq<tv;++tq){tn->l[tu+tq]=(l)tp;}}}return 1;}static void by(//////
int16_t*tn,bu*to){int tp;for(tp=0;tp<(1<<9);++tp){l tq=to->l[tp];tn[tp]=0;if(tq
<255){int tr=to->ab[tq];int ts=(tr>>4)&15;int tt=tr&15;int tu=to->ac[tq];if(tt
&&tu+tt<=9){int tv=((tp<<tu)&((1<<9)-1))>>(9-tt);int tw=1<<(tt-1);if(tv<tw)tv+=
(~0U<<tt)+1;if(tv>=-128&&tv<=127)tn[tp]=(int16_t)((tv*256)+(ts*16)+(tu+tt));}}}
}static void bz(bv*tn){do{unsigned int to=tn->ap?0:bk(tn->l);if(to==0xff){int//
tp=bk(tn->l);while(tp==0xff)tp=bk(tn->l);if(tp!=0){tn->ao=(unsigned char)tp;tn
->ap=1;return;}}tn->am|=to<<(24-tn->an);tn->an+=8;}while(tn->an<=24);}static///
const uint32_t ca[17]={0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,
32767,65535};static int cc(bv*tn,bu*to){unsigned int tp;int tq,tr;if(tn->an<16)
bz(tn);tq=(tn->am>>(32-9))&((1<<9)-1);tr=to->l[tq];if(tr<255){int ts=to->ac[tr]
;if(ts>tn->an)return-1;tn->am<<=ts;tn->an-=ts;return to->ab[tr];}tp=tn->am>>16;
for(tr=9+1;;++tr)if(tp<to->ad[tr])break;if(tr==17){tn->an-=16;return-1;}if(tr>
tn->an)return-1;tq=((tn->am>>(32-tr))&ca[tr])+to->ae[tr];if(tq<0||tq>=256)/////
return-1;assert((((tn->am)>>(32-to->ac[tq]))&ca[to->ac[tq]])==to->u[tq]);tn->an
-=tr;tn->am<<=tr;return to->ab[tq];}static const int cd[16]={0,-1,-3,-7,-15,-31
,-63,-127,-255,-511,-1023,-2047,-4095,-8191,-16383,-32767};static int ce(bv*tn,
int to){unsigned int tp;int tq;if(tn->an<to)bz(tn);if(tn->an<to)return 0;tq=tn
->am>>31;tp=(((tn->am)<<(to))|((tn->am)>>(-(to)&31)));tn->am=tp&~ca[to];tp&=ca[
to];tn->an-=to;return tp+(cd[to]&(tq-1));}static int cf(bv*tn,int to){unsigned
int tp;if(tn->an<to)bz(tn);if(tn->an<to)return 0;tp=(((tn->am)<<(to))|((tn->am)
>>(-(to)&31)));tn->am=tp&~ca[to];tp&=ca[to];tn->an-=to;return tp;}static int cg
(bv*tn){unsigned int to;if(tn->an<1)bz(tn);if(tn->an<1)return 0;to=tn->am;tn->
am<<=1;--tn->an;return to&0x80000000;}static const l ci[64+15]={0,1,8,16,9,2,3,
10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49
,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,
55,62,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63,63};static int cj(bv*tn,////
short to[64],bu*tp,bu*tq,int16_t*tr,int ts,uint16_t*tt){int tu,tv,tw;int tx;if(
tn->an<16)bz(tn);tx=cc(tn,tp);if(tx<0||tx>15)return ar("bad huffman code");////
memset(to,0,64*sizeof(to[0]));tu=tx?ce(tn,tx):0;if(!az(tn->al[ts].ag,tu))return
ar("bad delta");tv=tn->al[ts].ag+tu;tn->al[ts].ag=tv;if(!ba(tv,tt[0]))return ar
("can't merge dc and ac");to[0]=(short)(tv*tt[0]);tw=1;do{unsigned int ty;int//
tz,ua,ub;if(tn->an<16)bz(tn);tz=(tn->am>>(32-9))&((1<<9)-1);ua=tr[tz];if(ua){tw
+=(ua>>4)&15;ub=ua&15;if(ub>tn->an)return ar("bad huffman code");tn->am<<=ub;tn
->an-=ub;ty=ci[tw++];to[ty]=(short)((ua>>8)*tt[ty]);}else{int uc=cc(tn,tq);if(
uc<0)return ar("bad huffman code");ub=uc&15;ua=uc>>4;if(ub==0){if(uc!=0xf0)////
break;tw+=16;}else{tw+=ua;ty=ci[tw++];to[ty]=(short)(ce(tn,ub)*tt[ty]);}}}while
(tw<64);return 1;}static int ck(bv*tn,short to[64],bu*tp,int tq){int tr,ts;int
tt;if(tn->as!=0)return ar("can't merge dc and ac");if(tn->an<16)bz(tn);if(tn->
at==0){memset(to,0,64*sizeof(to[0]));tt=cc(tn,tp);if(tt<0||tt>15)return ar(////
"can't merge dc and ac");tr=tt?ce(tn,tt):0;if(!az(tn->al[tq].ag,tr))return ar(
"bad delta");ts=tn->al[tq].ag+tr;tn->al[tq].ag=ts;if(!ba(ts,1<<tn->au))return//
ar("can't merge dc and ac");to[0]=(short)(ts*(1<<tn->au));}else{if(cg(tn))to[0]
+=(short)(1<<tn->au);}return 1;}static int cl(bv*tn,short to[64],bu*tp,int16_t*
tq){int tr;if(tn->ar==0)return ar("can't merge dc and ac");if(tn->at==0){int ts
=tn->au;if(tn->av){--tn->av;return 1;}tr=tn->ar;do{unsigned int tt;int tu,tv,tw
;if(tn->an<16)bz(tn);tu=(tn->am>>(32-9))&((1<<9)-1);tv=tq[tu];if(tv){tr+=(tv>>4
)&15;tw=tv&15;if(tw>tn->an)return ar("bad huffman code");tn->am<<=tw;tn->an-=tw
;tt=ci[tr++];to[tt]=(short)((tv>>8)*(1<<ts));}else{int tx=cc(tn,tp);if(tx<0)///
return ar("bad huffman code");tw=tx&15;tv=tx>>4;if(tw==0){if(tv<15){tn->av=(1<<
tv);if(tv)tn->av+=cf(tn,tv);--tn->av;break;}tr+=16;}else{tr+=tv;tt=ci[tr++];to[
tt]=(short)(ce(tn,tw)*(1<<ts));}}}while(tr<=tn->as);}else{short ty=(short)(1<<
tn->au);if(tn->av){--tn->av;for(tr=tn->ar;tr<=tn->as;++tr){short*tz=&to[ci[tr]]
;if(*tz!=0)if(cg(tn))if((*tz&ty)==0){if(*tz>0)*tz+=ty;else*tz-=ty;}}}else{tr=tn
->ar;do{int ua,ub;int uc=cc(tn,tp);if(uc<0)return ar("bad huffman code");ub=uc&
15;ua=uc>>4;if(ub==0){if(ua<15){tn->av=(1<<ua)-1;if(ua)tn->av+=cf(tn,ua);ua=64;
}else{}}else{if(ub!=1)return ar("bad huffman code");if(cg(tn))ub=ty;else ub=-ty
;}while(tr<=tn->as){short*ud=&to[ci[tr++]];if(*ud!=0){if(cg(tn))if((*ud&ty)==0)
{if(*ud>0)*ud+=ty;else*ud-=ty;}}else{if(ua==0){*ud=(short)ub;break;}--ua;}}}///
while(tr<=tn->as);}}return 1;}static l cn(int tn){if((unsigned int)tn>255){if(
tn<0)return 0;if(tn>255)return 255;}return(l)tn;}static void co(l*tn,int to,///
short tp[64]){int tq,tr[64],*ts=tr;l*tt;short*tu=tp;for(tq=0;tq<8;++tq,++tu,++
ts){if(tu[8]==0&&tu[16]==0&&tu[24]==0&&tu[32]==0&&tu[40]==0&&tu[48]==0&&tu[56]
==0){int tv=tu[0]*4;ts[0]=ts[8]=ts[16]=ts[24]=ts[32]=ts[40]=ts[48]=ts[56]=tv;}
else{int tw,tx,ty,tz,ua,ub,uc,ud,ue,uf,ug,uh,ui;ub=tu[16];uc=tu[48];ua=(ub+uc)*
((int)(((0.5411961f)*4096+0.5)));ty=ua+uc*((int)(((-1.847759065f)*4096+0.5)));
tz=ua+ub*((int)(((0.765366865f)*4096+0.5)));ub=tu[0];uc=tu[32];tw=((ub+uc)*4096
);tx=((ub-uc)*4096);uf=tw+tz;ui=tw-tz;ug=tx+ty;uh=tx-ty;tw=tu[56];tx=tu[40];ty=
tu[24];tz=tu[8];uc=tw+ty;ud=tx+tz;ua=tw+tz;ub=tx+ty;ue=(uc+ud)*((int)(((///////
1.175875602f)*4096+0.5)));tw=tw*((int)(((0.298631336f)*4096+0.5)));tx=tx*((int)
(((2.053119869f)*4096+0.5)));ty=ty*((int)(((3.072711026f)*4096+0.5)));tz=tz*((
int)(((1.501321110f)*4096+0.5)));ua=ue+ua*((int)(((-0.899976223f)*4096+0.5)));
ub=ue+ub*((int)(((-2.562915447f)*4096+0.5)));uc=uc*((int)(((-1.961570560f)*4096
+0.5)));ud=ud*((int)(((-0.390180644f)*4096+0.5)));tz+=ua+ud;ty+=ub+uc;tx+=ub+ud
;tw+=ua+uc;uf+=512;ug+=512;uh+=512;ui+=512;ts[0]=(uf+tz)>>10;ts[56]=(uf-tz)>>10
;ts[8]=(ug+ty)>>10;ts[48]=(ug-ty)>>10;ts[16]=(uh+tx)>>10;ts[40]=(uh-tx)>>10;ts[
24]=(ui+tw)>>10;ts[32]=(ui-tw)>>10;}}for(tq=0,ts=tr,tt=tn;tq<8;++tq,ts+=8,tt+=
to){int uj,uk,ul,um,un,uo,up,uq,ur,us,ut,uu,uv;uo=ts[2];up=ts[6];un=(uo+up)*((
int)(((0.5411961f)*4096+0.5)));ul=un+up*((int)(((-1.847759065f)*4096+0.5)));um=
un+uo*((int)(((0.765366865f)*4096+0.5)));uo=ts[0];up=ts[4];uj=((uo+up)*4096);uk
=((uo-up)*4096);us=uj+um;uv=uj-um;ut=uk+ul;uu=uk-ul;uj=ts[7];uk=ts[5];ul=ts[3];
um=ts[1];up=uj+ul;uq=uk+um;un=uj+um;uo=uk+ul;ur=(up+uq)*((int)(((1.175875602f)*
4096+0.5)));uj=uj*((int)(((0.298631336f)*4096+0.5)));uk=uk*((int)(((///////////
2.053119869f)*4096+0.5)));ul=ul*((int)(((3.072711026f)*4096+0.5)));um=um*((int)
(((1.501321110f)*4096+0.5)));un=ur+un*((int)(((-0.899976223f)*4096+0.5)));uo=ur
+uo*((int)(((-2.562915447f)*4096+0.5)));up=up*((int)(((-1.961570560f)*4096+0.5)
));uq=uq*((int)(((-0.390180644f)*4096+0.5)));um+=un+uq;ul+=uo+up;uk+=uo+uq;uj+=
un+up;us+=65536+(128<<17);ut+=65536+(128<<17);uu+=65536+(128<<17);uv+=65536+(//
128<<17);tt[0]=cn((us+um)>>17);tt[7]=cn((us-um)>>17);tt[1]=cn((ut+ul)>>17);tt[6
]=cn((ut-ul)>>17);tt[2]=cn((uu+uk)>>17);tt[5]=cn((uu-uk)>>17);tt[3]=cn((uv+uj)
>>17);tt[4]=cn((uv-uj)>>17);}}static void cp(l*tn,int to,short tp[64]){__m128i
tq,tr,ts,tt,tu,tv,tw,tx;__m128i ty;__m128i tz=_mm_setr_epi16((((int)(((////////
0.5411961f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))+((int)(((-//////////
1.847759065f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))),(((int)(((///////
0.5411961f)*4096+0.5)))+((int)(((-1.847759065f)*4096+0.5)))),(((int)(((////////
0.5411961f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))+((int)(((-//////////
1.847759065f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))),(((int)(((///////
0.5411961f)*4096+0.5)))+((int)(((-1.847759065f)*4096+0.5)))));__m128i ua=//////
_mm_setr_epi16((((int)(((0.5411961f)*4096+0.5)))+((int)(((0.765366865f)*4096+//
0.5)))),(((int)(((0.5411961f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))+((
int)(((0.765366865f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))),(((int)(((
0.5411961f)*4096+0.5)))+((int)(((0.765366865f)*4096+0.5)))),(((int)(((/////////
0.5411961f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))+((int)(((///////////
0.765366865f)*4096+0.5)))),(((int)(((0.5411961f)*4096+0.5)))));__m128i ub=/////
_mm_setr_epi16((((int)(((1.175875602f)*4096+0.5)))+((int)(((-0.899976223f)*4096
+0.5)))),(((int)(((1.175875602f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)
))+((int)(((-0.899976223f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)))),((
(int)(((1.175875602f)*4096+0.5)))+((int)(((-0.899976223f)*4096+0.5)))),(((int)(
((1.175875602f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)))+((int)(((-////
0.899976223f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)))));__m128i uc=///
_mm_setr_epi16((((int)(((1.175875602f)*4096+0.5)))),(((int)(((1.175875602f)*///
4096+0.5)))+((int)(((-2.562915447f)*4096+0.5)))),(((int)(((1.175875602f)*4096+
0.5)))),(((int)(((1.175875602f)*4096+0.5)))+((int)(((-2.562915447f)*4096+0.5)))
),(((int)(((1.175875602f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)))+((//
int)(((-2.562915447f)*4096+0.5)))),(((int)(((1.175875602f)*4096+0.5)))),(((int)
(((1.175875602f)*4096+0.5)))+((int)(((-2.562915447f)*4096+0.5)))));__m128i ud=
_mm_setr_epi16((((int)(((-1.961570560f)*4096+0.5)))+((int)(((0.298631336f)*4096
+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+//
0.5)))+((int)(((0.298631336f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))
),(((int)(((-1.961570560f)*4096+0.5)))+((int)(((0.298631336f)*4096+0.5)))),(((
int)(((-1.961570560f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))+((int)(
((0.298631336f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))));__m128i ue=
_mm_setr_epi16((((int)(((-1.961570560f)*4096+0.5)))),(((int)(((-1.961570560f)*
4096+0.5)))+((int)(((3.072711026f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+
0.5)))),(((int)(((-1.961570560f)*4096+0.5)))+((int)(((3.072711026f)*4096+0.5)))
),(((int)(((-1.961570560f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))+((
int)(((3.072711026f)*4096+0.5)))),(((int)(((-1.961570560f)*4096+0.5)))),(((int)
(((-1.961570560f)*4096+0.5)))+((int)(((3.072711026f)*4096+0.5)))));__m128i uf=
_mm_setr_epi16((((int)(((-0.390180644f)*4096+0.5)))+((int)(((2.053119869f)*4096
+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+//
0.5)))+((int)(((2.053119869f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))
),(((int)(((-0.390180644f)*4096+0.5)))+((int)(((2.053119869f)*4096+0.5)))),(((
int)(((-0.390180644f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))+((int)(
((2.053119869f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))));__m128i ug=
_mm_setr_epi16((((int)(((-0.390180644f)*4096+0.5)))),(((int)(((-0.390180644f)*
4096+0.5)))+((int)(((1.501321110f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+
0.5)))),(((int)(((-0.390180644f)*4096+0.5)))+((int)(((1.501321110f)*4096+0.5)))
),(((int)(((-0.390180644f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))+((
int)(((1.501321110f)*4096+0.5)))),(((int)(((-0.390180644f)*4096+0.5)))),(((int)
(((-0.390180644f)*4096+0.5)))+((int)(((1.501321110f)*4096+0.5)))));__m128i uh=
_mm_set1_epi32(512);__m128i ui=_mm_set1_epi32(65536+(128<<17));tq=/////////////
_mm_load_si128((const __m128i*)(tp+0*8));tr=_mm_load_si128((const __m128i*)(tp+
1*8));ts=_mm_load_si128((const __m128i*)(tp+2*8));tt=_mm_load_si128((const/////
__m128i*)(tp+3*8));tu=_mm_load_si128((const __m128i*)(tp+4*8));tv=/////////////
_mm_load_si128((const __m128i*)(tp+5*8));tw=_mm_load_si128((const __m128i*)(tp+
6*8));tx=_mm_load_si128((const __m128i*)(tp+7*8));{__m128i uj=/////////////////
_mm_unpacklo_epi16((ts),(tw));__m128i uk=_mm_unpackhi_epi16((ts),(tw));__m128i
ul=_mm_madd_epi16(uj,tz);__m128i um=_mm_madd_epi16(uk,tz);__m128i un=//////////
_mm_madd_epi16(uj,ua);__m128i uo=_mm_madd_epi16(uk,ua);__m128i up=_mm_add_epi16
(tq,tu);__m128i uq=_mm_sub_epi16(tq,tu);__m128i ur=_mm_srai_epi32(/////////////
_mm_unpacklo_epi16(_mm_setzero_si128(),(up)),4);__m128i us=_mm_srai_epi32(/////
_mm_unpackhi_epi16(_mm_setzero_si128(),(up)),4);__m128i ut=_mm_srai_epi32(/////
_mm_unpacklo_epi16(_mm_setzero_si128(),(uq)),4);__m128i uu=_mm_srai_epi32(/////
_mm_unpackhi_epi16(_mm_setzero_si128(),(uq)),4);__m128i uv=_mm_add_epi32(ur,un)
;__m128i uw=_mm_add_epi32(us,uo);__m128i ux=_mm_sub_epi32(ur,un);__m128i uy=///
_mm_sub_epi32(us,uo);__m128i uz=_mm_add_epi32(ut,ul);__m128i va=_mm_add_epi32(
uu,um);__m128i vb=_mm_sub_epi32(ut,ul);__m128i vd=_mm_sub_epi32(uu,um);__m128i
ve=_mm_unpacklo_epi16((tx),(tt));__m128i vf=_mm_unpackhi_epi16((tx),(tt));/////
__m128i vg=_mm_madd_epi16(ve,ud);__m128i vh=_mm_madd_epi16(vf,ud);__m128i vi=//
_mm_madd_epi16(ve,ue);__m128i vj=_mm_madd_epi16(vf,ue);__m128i vk=/////////////
_mm_unpacklo_epi16((tv),(tr));__m128i vl=_mm_unpackhi_epi16((tv),(tr));__m128i
vm=_mm_madd_epi16(vk,uf);__m128i vn=_mm_madd_epi16(vl,uf);__m128i vo=//////////
_mm_madd_epi16(vk,ug);__m128i vp=_mm_madd_epi16(vl,ug);__m128i vq=_mm_add_epi16
(tr,tx);__m128i vr=_mm_add_epi16(tt,tv);__m128i vs=_mm_unpacklo_epi16((vq),(vr)
);__m128i vt=_mm_unpackhi_epi16((vq),(vr));__m128i vu=_mm_madd_epi16(vs,ub);///
__m128i vv=_mm_madd_epi16(vt,ub);__m128i vw=_mm_madd_epi16(vs,uc);__m128i vx=//
_mm_madd_epi16(vt,uc);__m128i vy=_mm_add_epi32(vg,vu);__m128i vz=_mm_add_epi32(
vh,vv);__m128i wa=_mm_add_epi32(vm,vw);__m128i wb=_mm_add_epi32(vn,vx);__m128i
wc=_mm_add_epi32(vi,vw);__m128i wd=_mm_add_epi32(vj,vx);__m128i we=////////////
_mm_add_epi32(vo,vu);__m128i wf=_mm_add_epi32(vp,vv);{__m128i wg=_mm_add_epi32(
uv,uh);__m128i wh=_mm_add_epi32(uw,uh);__m128i wi=_mm_add_epi32(wg,we);__m128i
wj=_mm_add_epi32(wh,wf);__m128i wk=_mm_sub_epi32(wg,we);__m128i wl=////////////
_mm_sub_epi32(wh,wf);tq=_mm_packs_epi32(_mm_srai_epi32(wi,10),_mm_srai_epi32(wj
,10));tx=_mm_packs_epi32(_mm_srai_epi32(wk,10),_mm_srai_epi32(wl,10));};{//////
__m128i wm=_mm_add_epi32(uz,uh);__m128i wn=_mm_add_epi32(va,uh);__m128i wo=////
_mm_add_epi32(wm,wc);__m128i wp=_mm_add_epi32(wn,wd);__m128i wq=_mm_sub_epi32(
wm,wc);__m128i wr=_mm_sub_epi32(wn,wd);tr=_mm_packs_epi32(_mm_srai_epi32(wo,10)
,_mm_srai_epi32(wp,10));tw=_mm_packs_epi32(_mm_srai_epi32(wq,10),_mm_srai_epi32
(wr,10));};{__m128i ws=_mm_add_epi32(vb,uh);__m128i wt=_mm_add_epi32(vd,uh);///
__m128i wu=_mm_add_epi32(ws,wa);__m128i wv=_mm_add_epi32(wt,wb);__m128i ww=////
_mm_sub_epi32(ws,wa);__m128i wx=_mm_sub_epi32(wt,wb);ts=_mm_packs_epi32(///////
_mm_srai_epi32(wu,10),_mm_srai_epi32(wv,10));tv=_mm_packs_epi32(_mm_srai_epi32(
ww,10),_mm_srai_epi32(wx,10));};{__m128i wy=_mm_add_epi32(ux,uh);__m128i wz=///
_mm_add_epi32(uy,uh);__m128i xa=_mm_add_epi32(wy,vy);__m128i xb=_mm_add_epi32(
wz,vz);__m128i xc=_mm_sub_epi32(wy,vy);__m128i xd=_mm_sub_epi32(wz,vz);tt=/////
_mm_packs_epi32(_mm_srai_epi32(xa,10),_mm_srai_epi32(xb,10));tu=_mm_packs_epi32
(_mm_srai_epi32(xc,10),_mm_srai_epi32(xd,10));};};{ty=tq;tq=_mm_unpacklo_epi16(
tq,tu);tu=_mm_unpackhi_epi16(ty,tu);ty=tr;tr=_mm_unpacklo_epi16(tr,tv);tv=/////
_mm_unpackhi_epi16(ty,tv);ty=ts;ts=_mm_unpacklo_epi16(ts,tw);tw=///////////////
_mm_unpackhi_epi16(ty,tw);ty=tt;tt=_mm_unpacklo_epi16(tt,tx);tx=///////////////
_mm_unpackhi_epi16(ty,tx);ty=tq;tq=_mm_unpacklo_epi16(tq,ts);ts=///////////////
_mm_unpackhi_epi16(ty,ts);ty=tr;tr=_mm_unpacklo_epi16(tr,tt);tt=///////////////
_mm_unpackhi_epi16(ty,tt);ty=tu;tu=_mm_unpacklo_epi16(tu,tw);tw=///////////////
_mm_unpackhi_epi16(ty,tw);ty=tv;tv=_mm_unpacklo_epi16(tv,tx);tx=///////////////
_mm_unpackhi_epi16(ty,tx);ty=tq;tq=_mm_unpacklo_epi16(tq,tr);tr=///////////////
_mm_unpackhi_epi16(ty,tr);ty=ts;ts=_mm_unpacklo_epi16(ts,tt);tt=///////////////
_mm_unpackhi_epi16(ty,tt);ty=tu;tu=_mm_unpacklo_epi16(tu,tv);tv=///////////////
_mm_unpackhi_epi16(ty,tv);ty=tw;tw=_mm_unpacklo_epi16(tw,tx);tx=///////////////
_mm_unpackhi_epi16(ty,tx);}{__m128i xe=_mm_unpacklo_epi16((ts),(tw));__m128i xf
=_mm_unpackhi_epi16((ts),(tw));__m128i xg=_mm_madd_epi16(xe,tz);__m128i xh=////
_mm_madd_epi16(xf,tz);__m128i xi=_mm_madd_epi16(xe,ua);__m128i xj=/////////////
_mm_madd_epi16(xf,ua);__m128i xk=_mm_add_epi16(tq,tu);__m128i xl=_mm_sub_epi16(
tq,tu);__m128i xm=_mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(),(xk)),4
);__m128i xn=_mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(),(xk)),4);///
__m128i xo=_mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(),(xl)),4);/////
__m128i xp=_mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(),(xl)),4);/////
__m128i xq=_mm_add_epi32(xm,xi);__m128i xr=_mm_add_epi32(xn,xj);__m128i xs=////
_mm_sub_epi32(xm,xi);__m128i xt=_mm_sub_epi32(xn,xj);__m128i xu=_mm_add_epi32(
xo,xg);__m128i xv=_mm_add_epi32(xp,xh);__m128i xx=_mm_sub_epi32(xo,xg);__m128i
xy=_mm_sub_epi32(xp,xh);__m128i xz=_mm_unpacklo_epi16((tx),(tt));__m128i ya=///
_mm_unpackhi_epi16((tx),(tt));__m128i yb=_mm_madd_epi16(xz,ud);__m128i yc=/////
_mm_madd_epi16(ya,ud);__m128i yd=_mm_madd_epi16(xz,ue);__m128i ye=/////////////
_mm_madd_epi16(ya,ue);__m128i yf=_mm_unpacklo_epi16((tv),(tr));__m128i yg=/////
_mm_unpackhi_epi16((tv),(tr));__m128i yh=_mm_madd_epi16(yf,uf);__m128i yi=/////
_mm_madd_epi16(yg,uf);__m128i yj=_mm_madd_epi16(yf,ug);__m128i yk=/////////////
_mm_madd_epi16(yg,ug);__m128i yl=_mm_add_epi16(tr,tx);__m128i ym=_mm_add_epi16(
tt,tv);__m128i yn=_mm_unpacklo_epi16((yl),(ym));__m128i yo=_mm_unpackhi_epi16((
yl),(ym));__m128i yp=_mm_madd_epi16(yn,ub);__m128i yq=_mm_madd_epi16(yo,ub);///
__m128i yr=_mm_madd_epi16(yn,uc);__m128i ys=_mm_madd_epi16(yo,uc);__m128i yt=//
_mm_add_epi32(yb,yp);__m128i yu=_mm_add_epi32(yc,yq);__m128i yv=_mm_add_epi32(
yh,yr);__m128i yx=_mm_add_epi32(yi,ys);__m128i yy=_mm_add_epi32(yd,yr);__m128i
yz=_mm_add_epi32(ye,ys);__m128i za=_mm_add_epi32(yj,yp);__m128i zb=////////////
_mm_add_epi32(yk,yq);{__m128i zc=_mm_add_epi32(xq,ui);__m128i zd=_mm_add_epi32(
xr,ui);__m128i ze=_mm_add_epi32(zc,za);__m128i zf=_mm_add_epi32(zd,zb);__m128i
zg=_mm_sub_epi32(zc,za);__m128i zh=_mm_sub_epi32(zd,zb);tq=_mm_packs_epi32(////
_mm_srai_epi32(ze,17),_mm_srai_epi32(zf,17));tx=_mm_packs_epi32(_mm_srai_epi32(
zg,17),_mm_srai_epi32(zh,17));};{__m128i zi=_mm_add_epi32(xu,ui);__m128i zj=///
_mm_add_epi32(xv,ui);__m128i zk=_mm_add_epi32(zi,yy);__m128i zl=_mm_add_epi32(
zj,yz);__m128i zm=_mm_sub_epi32(zi,yy);__m128i zn=_mm_sub_epi32(zj,yz);tr=/////
_mm_packs_epi32(_mm_srai_epi32(zk,17),_mm_srai_epi32(zl,17));tw=_mm_packs_epi32
(_mm_srai_epi32(zm,17),_mm_srai_epi32(zn,17));};{__m128i zo=_mm_add_epi32(xx,ui
);__m128i zp=_mm_add_epi32(xy,ui);__m128i zq=_mm_add_epi32(zo,yv);__m128i zr=//
_mm_add_epi32(zp,yx);__m128i zs=_mm_sub_epi32(zo,yv);__m128i zt=_mm_sub_epi32(
zp,yx);ts=_mm_packs_epi32(_mm_srai_epi32(zq,17),_mm_srai_epi32(zr,17));tv=/////
_mm_packs_epi32(_mm_srai_epi32(zs,17),_mm_srai_epi32(zt,17));};{__m128i zu=////
_mm_add_epi32(xs,ui);__m128i zv=_mm_add_epi32(xt,ui);__m128i zw=_mm_add_epi32(
zu,yt);__m128i zx=_mm_add_epi32(zv,yu);__m128i zy=_mm_sub_epi32(zu,yt);__m128i
zz=_mm_sub_epi32(zv,yu);tt=_mm_packs_epi32(_mm_srai_epi32(zw,17),_mm_srai_epi32
(zx,17));tu=_mm_packs_epi32(_mm_srai_epi32(zy,17),_mm_srai_epi32(zz,17));};};{
__m128i aaa=_mm_packus_epi16(tq,tr);__m128i aab=_mm_packus_epi16(ts,tt);__m128i
aac=_mm_packus_epi16(tu,tv);__m128i aad=_mm_packus_epi16(tw,tx);ty=aaa;aaa=////
_mm_unpacklo_epi8(aaa,aac);aac=_mm_unpackhi_epi8(ty,aac);ty=aab;aab=///////////
_mm_unpacklo_epi8(aab,aad);aad=_mm_unpackhi_epi8(ty,aad);ty=aaa;aaa=///////////
_mm_unpacklo_epi8(aaa,aab);aab=_mm_unpackhi_epi8(ty,aab);ty=aac;aac=///////////
_mm_unpacklo_epi8(aac,aad);aad=_mm_unpackhi_epi8(ty,aad);ty=aaa;aaa=///////////
_mm_unpacklo_epi8(aaa,aac);aac=_mm_unpackhi_epi8(ty,aac);ty=aab;aab=///////////
_mm_unpacklo_epi8(aab,aad);aad=_mm_unpackhi_epi8(ty,aad);_mm_storel_epi64((////
__m128i*)tn,aaa);tn+=to;_mm_storel_epi64((__m128i*)tn,_mm_shuffle_epi32(aaa,///
0x4e));tn+=to;_mm_storel_epi64((__m128i*)tn,aac);tn+=to;_mm_storel_epi64((/////
__m128i*)tn,_mm_shuffle_epi32(aac,0x4e));tn+=to;_mm_storel_epi64((__m128i*)tn,
aab);tn+=to;_mm_storel_epi64((__m128i*)tn,_mm_shuffle_epi32(aab,0x4e));tn+=to;
_mm_storel_epi64((__m128i*)tn,aad);tn+=to;_mm_storel_epi64((__m128i*)tn,///////
_mm_shuffle_epi32(aad,0x4e));}}static l cq(bv*tn){l to;if(tn->ao!=0xff){to=tn->
ao;tn->ao=0xff;return to;}to=bk(tn->l);if(to!=0xff)return 0xff;while(to==0xff)
to=bk(tn->l);return to;}static void ct(bv*tn){tn->an=0;tn->am=0;tn->ap=0;tn->al
[0].ag=tn->al[1].ag=tn->al[2].ag=tn->al[3].ag=0;tn->ao=0xff;tn->bc=tn->bb?tn->
bb:0x7fffffff;tn->av=0;}static int cu(bv*tn){ct(tn);if(!tn->aq){if(tn->az==1){
int to,tp;short tq[64]__attribute__((aligned(16)));int tr=tn->ba[0];int ts=(tn
->al[tr].ah+7)>>3;int tt=(tn->al[tr].ai+7)>>3;for(tp=0;tp<tt;++tp){for(to=0;to<
ts;++to){int tu=tn->al[tr].ae;if(!cj(tn,tq,tn->u+tn->al[tr].ad,tn->ab+tu,tn->ad
[tu],tr,tn->ac[tn->al[tr].ac]))return 0;tn->bd(tn->al[tr].al+tn->al[tr].aj*tp*8
+to*8,tn->al[tr].aj,tq);if(--tn->bc<=0){if(tn->an<24)bz(tn);if(!((tn->ao)>=0xd0
&&(tn->ao)<=0xd7))return 1;ct(tn);}}}return 1;}else{int tv,tw,tx,ty,tz;short ua
[64]__attribute__((aligned(16)));for(tw=0;tw<tn->ai;++tw){for(tv=0;tv<tn->ah;++
tv){for(tx=0;tx<tn->az;++tx){int ub=tn->ba[tx];for(tz=0;tz<tn->al[ub].ab;++tz){
for(ty=0;ty<tn->al[ub].u;++ty){int uc=(tv*tn->al[ub].u+ty)*8;int ud=(tw*tn->al[
ub].ab+tz)*8;int ue=tn->al[ub].ae;if(!cj(tn,ua,tn->u+tn->al[ub].ad,tn->ab+ue,tn
->ad[ue],ub,tn->ac[tn->al[ub].ac]))return 0;tn->bd(tn->al[ub].al+tn->al[ub].aj*
ud+uc,tn->al[ub].aj,ua);}}}if(--tn->bc<=0){if(tn->an<24)bz(tn);if(!((tn->ao)>=
0xd0&&(tn->ao)<=0xd7))return 1;ct(tn);}}}return 1;}}else{if(tn->az==1){int uf,
ug;int uh=tn->ba[0];int ui=(tn->al[uh].ah+7)>>3;int uj=(tn->al[uh].ai+7)>>3;for
(ug=0;ug<uj;++ug){for(uf=0;uf<ui;++uf){short*uk=tn->al[uh].ap+64*(uf+ug*tn->al[
uh].aq);if(tn->ar==0){if(!ck(tn,uk,&tn->u[tn->al[uh].ad],uh))return 0;}else{int
ul=tn->al[uh].ae;if(!cl(tn,uk,&tn->ab[ul],tn->ad[ul]))return 0;}if(--tn->bc<=0)
{if(tn->an<24)bz(tn);if(!((tn->ao)>=0xd0&&(tn->ao)<=0xd7))return 1;ct(tn);}}}//
return 1;}else{int um,un,uo,up,uq;for(un=0;un<tn->ai;++un){for(um=0;um<tn->ah;
++um){for(uo=0;uo<tn->az;++uo){int ur=tn->ba[uo];for(uq=0;uq<tn->al[ur].ab;++uq
){for(up=0;up<tn->al[ur].u;++up){int us=(um*tn->al[ur].u+up);int ut=(un*tn->al[
ur].ab+uq);short*uu=tn->al[ur].ap+64*(us+ut*tn->al[ur].aq);if(!ck(tn,uu,&tn->u[
tn->al[ur].ad],ur))return 0;}}}if(--tn->bc<=0){if(tn->an<24)bz(tn);if(!((tn->ao
)>=0xd0&&(tn->ao)<=0xd7))return 1;ct(tn);}}}return 1;}}}static void cv(short*tn
,uint16_t*to){int tp;for(tp=0;tp<64;++tp)tn[tp]*=to[tp];}static void cw(bv*tn){
if(tn->aq){int to,tp,tq;for(tq=0;tq<tn->l->ab;++tq){int tr=(tn->al[tq].ah+7)>>3
;int ts=(tn->al[tq].ai+7)>>3;for(tp=0;tp<ts;++tp){for(to=0;to<tr;++to){short*tt
=tn->al[tq].ap+64*(to+tp*tn->al[tq].aq);cv(tt,tn->ac[tn->al[tq].ac]);tn->bd(tn
->al[tq].al+tn->al[tq].aj*tp*8+to*8,tn->al[tq].aj,tt);}}}}}static int cx(bv*tn,
int to){int tp;switch(to){case 0xff:return ar("expected marker");case 0xDD:if(
bo(tn->l)!=4)return ar("bad DRI len");tn->bb=bo(tn->l);return 1;case 0xDB:tp=bo
(tn->l)-2;while(tp>0){int tq=bk(tn->l);int tr=tq>>4,ts=(tr!=0);int tt=tq&15,tu;
if(tr!=0&&tr!=1)return ar("bad DQT type");if(tt>3)return ar("bad DQT table");//
for(tu=0;tu<64;++tu)tn->ac[tt][ci[tu]]=(uint16_t)(ts?bo(tn->l):bk(tn->l));tp-=(
ts?129:65);}return tp==0;case 0xC4:tp=bo(tn->l)-2;while(tp>0){l*tv;int tw[16],
tx,ty=0;int tz=bk(tn->l);int ua=tz>>4;int ub=tz&15;if(ua>1||ub>3)return ar(////
"bad DHT header");for(tx=0;tx<16;++tx){tw[tx]=bk(tn->l);ty+=tw[tx];}if(ty>256)
return ar("bad DHT header");tp-=17;if(ua==0){if(!bx(tn->u+ub,tw))return 0;tv=tn
->u[ub].ab;}else{if(!bx(tn->ab+ub,tw))return 0;tv=tn->ab[ub].ab;}for(tx=0;tx<ty
;++tx)tv[tx]=bk(tn->l);if(ua!=0)by(tn->ad[ub],tn->ab+ub);tp-=ty;}return tp==0;}
if((to>=0xE0&&to<=0xEF)||to==0xFE){tp=bo(tn->l);if(tp<2){if(to==0xFE)return ar(
"bad COM len");else return ar("bad APP len");}tp-=2;if(to==0xE0&&tp>=5){static
const unsigned char uc[5]={'J','F','I','F','\0'};int ud=1;int ue;for(ue=0;ue<5;
++ue)if(bk(tn->l)!=uc[ue])ud=0;tp-=5;if(ud)tn->aw=1;}else if(to==0xEE&&tp>=12){
static const unsigned char uf[6]={'A','d','o','b','e','\0'};int ug=1;int uh;for
(uh=0;uh<6;++uh)if(bk(tn->l)!=uf[uh])ug=0;tp-=6;if(ug){bk(tn->l);bo(tn->l);bo(
tn->l);tn->ax=bk(tn->l);tp-=6;}}bm(tn->l,tp);return 1;}return ar(//////////////
"unknown marker");}static int cy(bv*tn){int to;int tp=bo(tn->l);tn->az=bk(tn->l
);if(tn->az<1||tn->az>4||tn->az>(int)tn->l->ab)return ar(//////////////////////
"bad SOS component count");if(tp!=6+2*tn->az)return ar("bad SOS len");for(to=0;
to<tn->az;++to){int tq=bk(tn->l),tr;int ts=bk(tn->l);for(tr=0;tr<tn->l->ab;++tr
)if(tn->al[tr].l==tq)break;if(tr==tn->l->ab)return 0;tn->al[tr].ad=ts>>4;if(tn
->al[tr].ad>3)return ar("bad DC huff");tn->al[tr].ae=ts&15;if(tn->al[tr].ae>3)
return ar("bad AC huff");tn->ba[to]=tr;}{int tt;tn->ar=bk(tn->l);tn->as=bk(tn->
l);tt=bk(tn->l);tn->at=(tt>>4);tn->au=(tt&15);if(tn->aq){if(tn->ar>63||tn->as>
63||tn->ar>tn->as||tn->at>13||tn->au>13)return ar("bad SOS");}else{if(tn->ar!=0
)return ar("bad SOS");if(tn->at!=0||tn->au!=0)return ar("bad SOS");tn->as=63;}}
return 1;}static int cz(bv*tn,int to,int tp){int tq;for(tq=0;tq<to;++tq){if(tn
->al[tq].am){free(tn->al[tq].am);tn->al[tq].am=NULL;tn->al[tq].al=NULL;}if(tn->
al[tq].an){free(tn->al[tq].an);tn->al[tq].an=0;tn->al[tq].ap=0;}if(tn->al[tq].
ao){free(tn->al[tq].ao);tn->al[tq].ao=NULL;}}return tp;}static int da(bv*tn,int
to){ab*tp=tn->l;int tq,tr,ts,tt,tu=1,tv=1,tw;tq=bo(tp);if(tq<11)return ar(/////
"bad SOF len");tr=bk(tp);if(tr!=8)return ar("only 8-bit");tp->u=bo(tp);if(tp->u
==0)return ar("no header height");tp->l=bo(tp);if(tp->l==0)return ar("0 width")
;if(tp->u>(1<<24))return ar("too large");if(tp->l>(1<<24))return ar("too large"
);tw=bk(tp);if(tw!=3&&tw!=1&&tw!=4)return ar("bad component count");tp->ab=tw;
for(ts=0;ts<tw;++ts){tn->al[ts].al=NULL;tn->al[ts].ao=NULL;}if(tq!=8+3*tp->ab)
return ar("bad SOF len");tn->ay=0;for(ts=0;ts<tp->ab;++ts){static const////////
unsigned char tx[3]={'R','G','B'};tn->al[ts].l=bk(tp);if(tp->ab==3&&tn->al[ts].
l==tx[ts])++tn->ay;tt=bk(tp);tn->al[ts].u=(tt>>4);if(!tn->al[ts].u||tn->al[ts].
u>4)return ar("bad H");tn->al[ts].ab=tt&15;if(!tn->al[ts].ab||tn->al[ts].ab>4)
return ar("bad V");tn->al[ts].ac=bk(tp);if(tn->al[ts].ac>3)return ar("bad TQ");
}if(to!=STBI__SCAN_load)return 1;if(!aw(tp->l,tp->u,tp->ab,0))return ar(///////
"too large");for(ts=0;ts<tp->ab;++ts){if(tn->al[ts].u>tu)tu=tn->al[ts].u;if(tn
->al[ts].ab>tv)tv=tn->al[ts].ab;}for(ts=0;ts<tp->ab;++ts){if(tu%tn->al[ts].u!=0
)return ar("bad H");if(tv%tn->al[ts].ab!=0)return ar("bad V");}tn->ae=tu;tn->ag
=tv;tn->aj=tu*8;tn->ak=tv*8;tn->ah=(tp->l+tn->aj-1)/tn->aj;tn->ai=(tp->u+tn->ak
-1)/tn->ak;for(ts=0;ts<tp->ab;++ts){tn->al[ts].ah=(tp->l*tn->al[ts].u+tu-1)/tu;
tn->al[ts].ai=(tp->u*tn->al[ts].ab+tv-1)/tv;tn->al[ts].aj=tn->ah*tn->al[ts].u*8
;tn->al[ts].ak=tn->ai*tn->al[ts].ab*8;tn->al[ts].ap=0;tn->al[ts].an=0;tn->al[ts
].ao=NULL;tn->al[ts].am=ax(tn->al[ts].aj,tn->al[ts].ak,15);if(tn->al[ts].am==//
NULL)return cz(tn,ts+1,ar("outofmem"));tn->al[ts].al=(l*)(((size_t)tn->al[ts].
am+15)&~15);if(tn->aq){tn->al[ts].aq=tn->al[ts].aj/8;tn->al[ts].ar=tn->al[ts].
ak/8;tn->al[ts].an=ay(tn->al[ts].aj,tn->al[ts].ak,sizeof(short),15);if(tn->al[
ts].an==NULL)return cz(tn,ts+1,ar("outofmem"));tn->al[ts].ap=(short*)(((size_t)
tn->al[ts].an+15)&~15);}}return 1;}static int db(bv*tn,int to){int tp;tn->aw=0;
tn->ax=-1;tn->ao=0xff;tp=cq(tn);if(!((tp)==0xd8))return ar("no SOI");if(to==///
STBI__SCAN_type)return 1;tp=cq(tn);while(!((tp)==0xc0||(tp)==0xc1||(tp)==0xc2))
{if(!cx(tn,tp))return 0;tp=cq(tn);while(tp==0xff){if(bl(tn->l))return ar(//////
"no SOF");tp=cq(tn);}}tn->aq=((tp)==0xc2);if(!da(tn,to))return 0;return 1;}////
static l dd(bv*tn){while(!bl(tn->l)){l to=bk(tn->l);while(to==0xff){if(bl(tn->l
))return 0xff;to=bk(tn->l);if(to!=0x00&&to!=0xff){return to;}}}return 0xff;}///
static int de(bv*tn){int to;for(to=0;to<4;to++){tn->al[to].am=NULL;tn->al[to].
an=NULL;}tn->bb=0;if(!db(tn,STBI__SCAN_load))return 0;to=cq(tn);while(!((to)==
0xd9)){if((to)==0xda){if(!cy(tn))return 0;if(!cu(tn))return 0;if(tn->ao==0xff){
tn->ao=dd(tn);}to=cq(tn);if(((to)>=0xd0&&(to)<=0xd7))to=cq(tn);}else if((to)==
0xdc){int tp=bo(tn->l);uint32_t tq=bo(tn->l);if(tp!=4)return ar("bad DNL len");
if(tq!=tn->l->u)return ar("bad DNL height");to=cq(tn);}else{if(!cx(tn,to))/////
return 1;to=cq(tn);}}if(tn->aq)cw(tn);return 1;}typedef l*(*df)(l*out,l*in0,l*
in1,int w,int hs);static l*dg(l*tn,l*to,l*tp,int tq,int tr){(void)sizeof(tn);(
void)sizeof(tp);(void)sizeof(tq);(void)sizeof(tr);return to;}static l*dh(l*tn,l
*to,l*tp,int tq,int tr){int ts;(void)sizeof(tr);for(ts=0;ts<tq;++ts)tn[ts]=((l)
((3*to[ts]+tp[ts]+2)>>2));return tn;}static l*di(l*tn,l*to,l*tp,int tq,int tr){
int ts;l*tt=to;if(tq==1){tn[0]=tn[1]=tt[0];return tn;}tn[0]=tt[0];tn[1]=((l)((
tt[0]*3+tt[1]+2)>>2));for(ts=1;ts<tq-1;++ts){int tu=3*tt[ts]+2;tn[ts*2+0]=((l)(
(tu+tt[ts-1])>>2));tn[ts*2+1]=((l)((tu+tt[ts+1])>>2));}tn[ts*2+0]=((l)((tt[tq-2
]*3+tt[tq-1]+2)>>2));tn[ts*2+1]=tt[tq-1];(void)sizeof(tp);(void)sizeof(tr);////
return tn;}static l*dj(l*tn,l*to,l*tp,int tq,int tr){int ts,tt,tu;if(tq==1){tn[
0]=tn[1]=((l)((3*to[0]+tp[0]+2)>>2));return tn;}tu=3*to[0]+tp[0];tn[0]=((l)((tu
+2)>>2));for(ts=1;ts<tq;++ts){tt=tu;tu=3*to[ts]+tp[ts];tn[ts*2-1]=((l)((3*tt+tu
+8)>>4));tn[ts*2]=((l)((3*tu+tt+8)>>4));}tn[tq*2-1]=((l)((tu+2)>>2));(void)////
sizeof(tr);return tn;}static l*dk(l*tn,l*to,l*tp,int tq,int tr){int ts=0,tt,tu;
if(tq==1){tn[0]=tn[1]=((l)((3*to[0]+tp[0]+2)>>2));return tn;}tu=3*to[0]+tp[0];
for(;ts<((tq-1)&~7);ts+=8){__m128i tv=_mm_setzero_si128();__m128i tw=//////////
_mm_loadl_epi64((__m128i*)(tp+ts));__m128i tx=_mm_loadl_epi64((__m128i*)(to+ts)
);__m128i ty=_mm_unpacklo_epi8(tw,tv);__m128i tz=_mm_unpacklo_epi8(tx,tv);/////
__m128i ua=_mm_sub_epi16(ty,tz);__m128i ub=_mm_slli_epi16(tz,2);__m128i uc=////
_mm_add_epi16(ub,ua);__m128i ud=_mm_slli_si128(uc,2);__m128i ue=_mm_srli_si128(
uc,2);__m128i uf=_mm_insert_epi16(ud,tu,0);__m128i ug=_mm_insert_epi16(ue,3*to[
ts+8]+tp[ts+8],7);__m128i uh=_mm_set1_epi16(8);__m128i ui=_mm_slli_epi16(uc,2);
__m128i uj=_mm_sub_epi16(uf,uc);__m128i uk=_mm_sub_epi16(ug,uc);__m128i ul=////
_mm_add_epi16(ui,uh);__m128i um=_mm_add_epi16(uj,ul);__m128i un=_mm_add_epi16(
uk,ul);__m128i uo=_mm_unpacklo_epi16(um,un);__m128i up=_mm_unpackhi_epi16(um,un
);__m128i uq=_mm_srli_epi16(uo,4);__m128i ur=_mm_srli_epi16(up,4);__m128i us=//
_mm_packus_epi16(uq,ur);_mm_storeu_si128((__m128i*)(tn+ts*2),us);tu=3*to[ts+7]+
tp[ts+7];}tt=tu;tu=3*to[ts]+tp[ts];tn[ts*2]=((l)((3*tu+tt+8)>>4));for(++ts;ts<
tq;++ts){tt=tu;tu=3*to[ts]+tp[ts];tn[ts*2-1]=((l)((3*tt+tu+8)>>4));tn[ts*2]=((l
)((3*tu+tt+8)>>4));}tn[tq*2-1]=((l)((tu+2)>>2));(void)sizeof(tr);return tn;}///
static l*dl(l*tn,l*to,l*tp,int tq,int tr){int ts,tt;(void)sizeof(tp);for(ts=0;
ts<tq;++ts)for(tt=0;tt<tr;++tt)tn[ts*tr+tt]=to[ts];return tn;}static void dm(l*
tn,const l*to,const l*tp,const l*tq,int tr,int ts){int tt;for(tt=0;tt<tr;++tt){
int tu=(to[tt]<<20)+(1<<19);int tv,tw,tx;int ty=tq[tt]-128;int tz=tp[tt]-128;tv
=tu+ty*(((int)((1.40200f)*4096.0f+0.5f))<<8);tw=tu+(ty*-(((int)((0.71414f)*////
4096.0f+0.5f))<<8))+((tz*-(((int)((0.34414f)*4096.0f+0.5f))<<8))&0xffff0000);tx
=tu+tz*(((int)((1.77200f)*4096.0f+0.5f))<<8);tv>>=20;tw>>=20;tx>>=20;if((//////
unsigned)tv>255){if(tv<0)tv=0;else tv=255;}if((unsigned)tw>255){if(tw<0)tw=0;//
else tw=255;}if((unsigned)tx>255){if(tx<0)tx=0;else tx=255;}tn[0]=(l)tv;tn[1]=(
l)tw;tn[2]=(l)tx;tn[3]=255;tn+=ts;}}static void dn(l*tn,l const*to,l const*tp,l
const*tq,int tr,int ts){int tt=0;if(ts==4){__m128i tu=_mm_set1_epi8(-0x80);////
__m128i tv=_mm_set1_epi16((short)(1.40200f*4096.0f+0.5f));__m128i tw=//////////
_mm_set1_epi16(-(short)(0.71414f*4096.0f+0.5f));__m128i tx=_mm_set1_epi16(-(///
short)(0.34414f*4096.0f+0.5f));__m128i ty=_mm_set1_epi16((short)(1.77200f*/////
4096.0f+0.5f));__m128i tz=_mm_set1_epi8((char)(unsigned char)128);__m128i ua=//
_mm_set1_epi16(255);for(;tt+7<tr;tt+=8){__m128i ub=_mm_loadl_epi64((__m128i*)(
to+tt));__m128i uc=_mm_loadl_epi64((__m128i*)(tq+tt));__m128i ud=//////////////
_mm_loadl_epi64((__m128i*)(tp+tt));__m128i ue=_mm_xor_si128(uc,tu);__m128i uf=
_mm_xor_si128(ud,tu);__m128i ug=_mm_unpacklo_epi8(tz,ub);__m128i uh=///////////
_mm_unpacklo_epi8(_mm_setzero_si128(),ue);__m128i ui=_mm_unpacklo_epi8(////////
_mm_setzero_si128(),uf);__m128i uj=_mm_srli_epi16(ug,4);__m128i uk=////////////
_mm_mulhi_epi16(tv,uh);__m128i ul=_mm_mulhi_epi16(tx,ui);__m128i um=///////////
_mm_mulhi_epi16(ui,ty);__m128i un=_mm_mulhi_epi16(uh,tw);__m128i uo=///////////
_mm_add_epi16(uk,uj);__m128i up=_mm_add_epi16(ul,uj);__m128i uq=_mm_add_epi16(
uj,um);__m128i ur=_mm_add_epi16(up,un);__m128i us=_mm_srai_epi16(uo,4);__m128i
ut=_mm_srai_epi16(uq,4);__m128i uu=_mm_srai_epi16(ur,4);__m128i uv=////////////
_mm_packus_epi16(us,ut);__m128i uw=_mm_packus_epi16(uu,ua);__m128i ux=/////////
_mm_unpacklo_epi8(uv,uw);__m128i uy=_mm_unpackhi_epi8(uv,uw);__m128i uz=///////
_mm_unpacklo_epi16(ux,uy);__m128i va=_mm_unpackhi_epi16(ux,uy);_mm_storeu_si128
((__m128i*)(tn+0),uz);_mm_storeu_si128((__m128i*)(tn+16),va);tn+=32;}}for(;tt<
tr;++tt){int vb=(to[tt]<<20)+(1<<19);int vd,ve,vf;int vg=tq[tt]-128;int vh=tp[
tt]-128;vd=vb+vg*(((int)((1.40200f)*4096.0f+0.5f))<<8);ve=vb+vg*-(((int)((/////
0.71414f)*4096.0f+0.5f))<<8)+((vh*-(((int)((0.34414f)*4096.0f+0.5f))<<8))&/////
0xffff0000);vf=vb+vh*(((int)((1.77200f)*4096.0f+0.5f))<<8);vd>>=20;ve>>=20;vf//
>>=20;if((unsigned)vd>255){if(vd<0)vd=0;else vd=255;}if((unsigned)ve>255){if(ve
<0)ve=0;else ve=255;}if((unsigned)vf>255){if(vf<0)vf=0;else vf=255;}tn[0]=(l)vd
;tn[1]=(l)ve;tn[2]=(l)vf;tn[3]=255;tn+=ts;}}static void dp(bv*tn){tn->bd=co;tn
->be=dm;tn->bg=dj;tn->bd=cp;tn->be=dn;tn->bg=dk;}static void dq(bv*tn){cz(tn,tn
->l->ab,0);}typedef struct{df l;l*u,*ab;int ac,ad;int ae;int ag;int ah;}dr;////
static l ds(l tn,l to){unsigned int tp=tn*to+128;return(l)((tp+(tp>>8))>>8);}//
static l*dt(bv*tn,int*to,int*tp,int*tq,int tr){int ts,tt,tu;tn->l->ab=0;if(tr<0
||tr>4)return((unsigned char*)(size_t)(ar("bad req_comp")?NULL:NULL));if(!de(tn
)){dq(tn);return NULL;}ts=tr?tr:tn->l->ab>=3?3:1;tu=tn->l->ab==3&&(tn->ay==3||(
tn->ax==0&&!tn->aw));if(tn->l->ab==3&&ts<3&&!tu)tt=1;else tt=tn->l->ab;if(tt<=0
){dq(tn);return NULL;}{int tv;unsigned int tw,tx;l*ty;l*tz[4]={NULL,NULL,NULL,
NULL};dr ua[4];for(tv=0;tv<tt;++tv){dr*ub=&ua[tv];tn->al[tv].ao=(l*)as(tn->l->l
+3);if(!tn->al[tv].ao){dq(tn);return((unsigned char*)(size_t)(ar("outofmem")?//
NULL:NULL));}ub->ac=tn->ae/tn->al[tv].u;ub->ad=tn->ag/tn->al[tv].ab;ub->ag=ub->
ad>>1;ub->ae=(tn->l->l+ub->ac-1)/ub->ac;ub->ah=0;ub->u=ub->ab=tn->al[tv].al;if(
ub->ac==1&&ub->ad==1)ub->l=dg;else if(ub->ac==1&&ub->ad==2)ub->l=dh;else if(ub
->ac==2&&ub->ad==1)ub->l=di;else if(ub->ac==2&&ub->ad==2)ub->l=tn->bg;else ub->
l=dl;}ty=(l*)ay(ts,tn->l->l,tn->l->u,1);if(!ty){dq(tn);return((unsigned char*)(
size_t)(ar("outofmem")?NULL:NULL));}for(tx=0;tx<tn->l->u;++tx){l*uc=ty+ts*tn->l
->l*tx;for(tv=0;tv<tt;++tv){dr*ud=&ua[tv];int ue=ud->ag>=(ud->ad>>1);tz[tv]=ud
->l(tn->al[tv].ao,ue?ud->ab:ud->u,ue?ud->u:ud->ab,ud->ae,ud->ac);if(++ud->ag>=
ud->ad){ud->ag=0;ud->u=ud->ab;if(++ud->ah<tn->al[tv].ai)ud->ab+=tn->al[tv].aj;}
}if(ts>=3){l*uf=tz[0];if(tn->l->ab==3){if(tu){for(tw=0;tw<tn->l->l;++tw){uc[0]=
uf[tw];uc[1]=tz[1][tw];uc[2]=tz[2][tw];uc[3]=255;uc+=ts;}}else{tn->be(uc,uf,tz[
1],tz[2],tn->l->l,ts);}}else if(tn->l->ab==4){if(tn->ax==0){for(tw=0;tw<tn->l->
l;++tw){l ug=tz[3][tw];uc[0]=ds(tz[0][tw],ug);uc[1]=ds(tz[1][tw],ug);uc[2]=ds(
tz[2][tw],ug);uc[3]=255;uc+=ts;}}else if(tn->ax==2){tn->be(uc,uf,tz[1],tz[2],tn
->l->l,ts);for(tw=0;tw<tn->l->l;++tw){l uh=tz[3][tw];uc[0]=ds(255-uc[0],uh);uc[
1]=ds(255-uc[1],uh);uc[2]=ds(255-uc[2],uh);uc+=ts;}}else{tn->be(uc,uf,tz[1],tz[
2],tn->l->l,ts);}}else for(tw=0;tw<tn->l->l;++tw){uc[0]=uc[1]=uc[2]=uf[tw];uc[3
]=255;uc+=ts;}}else{if(tu){if(ts==1)for(tw=0;tw<tn->l->l;++tw)*uc++ =bq(tz[0][
tw],tz[1][tw],tz[2][tw]);else{for(tw=0;tw<tn->l->l;++tw,uc+=2){uc[0]=bq(tz[0][
tw],tz[1][tw],tz[2][tw]);uc[1]=255;}}}else if(tn->l->ab==4&&tn->ax==0){for(tw=0
;tw<tn->l->l;++tw){l ui=tz[3][tw];l uj=ds(tz[0][tw],ui);l uk=ds(tz[1][tw],ui);l
ul=ds(tz[2][tw],ui);uc[0]=bq(uj,uk,ul);uc[1]=255;uc+=ts;}}else if(tn->l->ab==4
&&tn->ax==2){for(tw=0;tw<tn->l->l;++tw){uc[0]=ds(255-tz[0][tw],tz[3][tw]);uc[1]
=255;uc+=ts;}}else{l*um=tz[0];if(ts==1)for(tw=0;tw<tn->l->l;++tw)uc[tw]=um[tw];
else for(tw=0;tw<tn->l->l;++tw){*uc++ =um[tw];*uc++ =255;}}}}dq(tn);*to=tn->l->
l;*tp=tn->l->u;if(tq)*tq=tn->l->ab>=3?3:1;return ty;}}static void*an(ab*tt,int*
tu,int*tv,int*tw,int tx,al*ty){unsigned char*tz;bv*ua=(bv*)as(sizeof(bv));if(!
ua)return((unsigned char*)(size_t)(ar("outofmem")?NULL:NULL));memset(ua,0,/////
sizeof(bv));(void)ty;ua->l=tt;dp(ua);tz=dt(ua,tu,tv,tw,tx);free(ua);return tz;}
static int am(ab*to){int tp;bv*tq=(bv*)as(sizeof(bv));if(!tq)return ar(////////
"outofmem");memset(tq,0,sizeof(bv));tq->l=to;dp(tq);tp=db(tq,STBI__SCAN_type);
ak(to);free(tq);return tp;}typedef struct{uint16_t l[1<<9];uint16_t u[16];int//
ab[17];uint16_t ac[16];l ad[288];uint16_t ae[288];}du;static int dv(int tn){tn=
((tn&0xAAAA)>>1)|((tn&0x5555)<<1);tn=((tn&0xCCCC)>>2)|((tn&0x3333)<<2);tn=((tn&
0xF0F0)>>4)|((tn&0x0F0F)<<4);tn=((tn&0xFF00)>>8)|((tn&0x00FF)<<8);return tn;}//
static int dw(int tn,int to){assert(to<=16);return dv(tn)>>(16-to);}static int
dx(du*tn,const l*to,int tp){int tq,tr=0;int ts,tt[16],tu[17];memset(tu,0,sizeof
(tu));memset(tn->l,0,sizeof(tn->l));for(tq=0;tq<tp;++tq)++tu[to[tq]];tu[0]=0;//
for(tq=1;tq<16;++tq)if(tu[tq]>(1<<tq))return ar("bad sizes");ts=0;for(tq=1;tq<
16;++tq){tt[tq]=ts;tn->u[tq]=(uint16_t)ts;tn->ac[tq]=(uint16_t)tr;ts=(ts+tu[tq]
);if(tu[tq])if(ts-1>=(1<<tq))return ar("bad codelengths");tn->ab[tq]=ts<<(16-tq
);ts<<=1;tr+=tu[tq];}tn->ab[16]=0x10000;for(tq=0;tq<tp;++tq){int tv=to[tq];if(
tv){int tw=tt[tv]-tn->u[tv]+tn->ac[tv];uint16_t tx=(uint16_t)((tv<<9)|tq);tn->
ad[tw]=(l)tv;tn->ae[tw]=(uint16_t)tq;if(tv<=9){int ty=dw(tt[tv],tv);while(ty<(1
<<9)){tn->l[ty]=tx;ty+=(1<<tv);}}++tt[tv];}}return 1;}typedef struct{l*l,*u;int
ab;int ac;uint32_t ad;char*ae;char*ag;char*ah;int ai;du aj,ak;}dy;static int dz
(dy*tn){return(tn->l>=tn->u);}static l ea(dy*tn){return dz(tn)?0:*tn->l++;}////
static void eb(dy*tn){do{if(tn->ad>=(1U<<tn->ab)){tn->l=tn->u;return;}tn->ad|=(
unsigned int)ea(tn)<<tn->ab;tn->ab+=8;}while(tn->ab<=24);}static unsigned int//
ec(dy*tn,int to){unsigned int tp;if(tn->ab<to)eb(tn);tp=tn->ad&((1<<to)-1);tn->
ad>>=to;tn->ab-=to;return tp;}static int ed(dy*tn,du*to){int tp,tq,tr;tr=dw(tn
->ad,16);for(tq=9+1;;++tq)if(tr<to->ab[tq])break;if(tq>=16)return-1;tp=(tr>>(16
-tq))-to->u[tq]+to->ac[tq];if(tp>=288)return-1;if(to->ad[tp]!=tq)return-1;tn->
ad>>=tq;tn->ab-=tq;return to->ae[tp];}static int ee(dy*tn,du*to){int tp,tq;if(
tn->ab<16){if(dz(tn)){if(!tn->ac){tn->ac=1;tn->ab+=16;}else{return-1;}}else{eb(
tn);}}tp=to->l[tn->ad&((1<<9)-1)];if(tp){tq=tp>>9;tn->ad>>=tq;tn->ab-=tq;return
tp&511;}return ed(tn,to);}static int ef(dy*tn,char*to,int tp){char*tq;unsigned
int tr,ts,tt;tn->ae=to;if(!tn->ai)return ar("output buffer limit");tr=(unsigned
int)(tn->ae-tn->ag);ts=tt=(unsigned)(tn->ah-tn->ag);if(UINT_MAX-tr<(unsigned)tp
)return ar("outofmem");while(tr+tp>ts){if(ts>UINT_MAX/2)return ar("outofmem");
ts*=2;}tq=(char*)realloc(tn->ag,ts);(void)sizeof(tt);if(tq==NULL)return ar(////
"outofmem");tn->ag=tq;tn->ae=tq+tr;tn->ah=tq+ts;return 1;}static const int eg[
31]={3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,
195,227,258,0,0};static const int eh[31]={0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3
,3,4,4,4,4,5,5,5,5,0,0,0};static const int ei[32]={1,2,3,4,5,7,9,13,17,25,33,49
,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,
24577,0,0};static const int ej[32]={0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9
,10,10,11,11,12,12,13,13};static int ek(dy*tn){char*to=tn->ae;for(;;){int tp=ee
(tn,&tn->aj);if(tp<256){if(tp<0)return ar("bad huffman code");if(to>=tn->ah){if
(!ef(tn,to,1))return 0;to=tn->ae;}*to++ =(char)tp;}else{l*tq;int tr,ts;if(tp==
256){tn->ae=to;if(tn->ac&&tn->ab<16){return ar("unexpected end");}return 1;}if(
tp>=286)return ar("bad huffman code");tp-=257;tr=eg[tp];if(eh[tp])tr+=ec(tn,eh[
tp]);tp=ee(tn,&tn->ak);if(tp<0||tp>=30)return ar("bad huffman code");ts=ei[tp];
if(ej[tp])ts+=ec(tn,ej[tp]);if(to-tn->ag<ts)return ar("bad dist");if(tr>tn->ah-
to){if(!ef(tn,to,tr))return 0;to=tn->ae;}tq=(l*)(to-ts);if(ts==1){l tt=*tq;if(
tr){do*to++ =tt;while(--tr);}}else{if(tr){do*to++ =*tq++;while(--tr);}}}}}/////
static int el(dy*tn){static const l to[19]={16,17,18,0,8,7,9,6,10,5,11,4,12,3,
13,2,14,1,15};du tp;l tq[286+32+137];l tr[19];int ts,tt;int tu=ec(tn,5)+257;int
tv=ec(tn,5)+1;int tw=ec(tn,4)+4;int tx=tu+tv;memset(tr,0,sizeof(tr));for(ts=0;
ts<tw;++ts){int ty=ec(tn,3);tr[to[ts]]=(l)ty;}if(!dx(&tp,tr,19))return 0;tt=0;
while(tt<tx){int tz=ee(tn,&tp);if(tz<0||tz>=19)return ar("bad codelengths");if(
tz<16)tq[tt++]=(l)tz;else{l ua=0;if(tz==16){tz=ec(tn,2)+3;if(tt==0)return ar(//
"bad codelengths");ua=tq[tt-1];}else if(tz==17){tz=ec(tn,3)+3;}else if(tz==18){
tz=ec(tn,7)+11;}else{return ar("bad codelengths");}if(tx-tt<tz)return ar(//////
"bad codelengths");memset(tq+tt,ua,tz);tt+=tz;}}if(tt!=tx)return ar(///////////
"bad codelengths");if(!dx(&tn->aj,tq,tu))return 0;if(!dx(&tn->ak,tq+tu,tv))////
return 0;return 1;}static int em(dy*tn){l to[4];int tp,tq,tr;if(tn->ab&7)ec(tn,
tn->ab&7);tr=0;while(tn->ab>0){to[tr++]=(l)(tn->ad&255);tn->ad>>=8;tn->ab-=8;}
if(tn->ab<0)return ar("zlib corrupt");while(tr<4)to[tr++]=ea(tn);tp=to[1]*256+
to[0];tq=to[3]*256+to[2];if(tq!=(tp^0xffff))return ar("zlib corrupt");if(tn->l+
tp>tn->u)return ar("read past buffer");if(tn->ae+tp>tn->ah)if(!ef(tn,tn->ae,tp)
)return 0;memcpy(tn->ae,tn->l,tp);tn->l+=tp;tn->ae+=tp;return 1;}static int en(
dy*tn){int to=ea(tn);int tp=to&15;int tq=ea(tn);if(dz(tn))return ar(///////////
"bad zlib header");if((to*256+tq)%31!=0)return ar("bad zlib header");if(tq&32)
return ar("no preset dict");if(tp!=8)return ar("bad compression");return 1;}///
static const l eo[288]={8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8
,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7
,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8};static const l ep[32]={5,5,5,5,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};static int eq(dy*tn,//
int to){int tp,tq;if(to)if(!en(tn))return 0;tn->ab=0;tn->ad=0;tn->ac=0;do{tp=ec
(tn,1);tq=ec(tn,2);if(tq==0){if(!em(tn))return 0;}else if(tq==3){return 0;}else
{if(tq==1){if(!dx(&tn->aj,eo,288))return 0;if(!dx(&tn->ak,ep,32))return 0;}else
{if(!el(tn))return 0;}if(!ek(tn))return 0;}}while(!tp);return 1;}static int er(
dy*tn,char*to,int tp,int tq,int tr){tn->ag=to;tn->ae=to;tn->ah=to+tp;tn->ai=tq;
return eq(tn,tr);}extern char*stbi_zlib_decode_malloc_guesssize_headerflag(////
const char*tn,int to,int tp,int*tq,int tr){dy ts;char*tt=(char*)as(tp);if(tt==
NULL)return NULL;ts.l=(l*)tn;ts.u=(l*)tn+to;if(er(&ts,tt,tp,1,tr)){if(tq)*tq=(
int)(ts.ae-ts.ag);return ts.ag;}else{free(ts.ag);return NULL;}}typedef struct{
uint32_t l;uint32_t u;}es;static es et(ab*tn){es to;to.l=bp(tn);to.u=bp(tn);///
return to;}static int eu(ab*tn){static const l to[8]={137,80,78,71,13,10,26,10}
;int tp;for(tp=0;tp<8;++tp)if(bk(tn)!=to[tp])return ar("bad png sig");return 1;
}typedef struct{ab*l;l*u,*ab,*ac;int ad;}ev;enum{STBI__F_none=0,STBI__F_sub=1,
STBI__F_up=2,STBI__F_avg=3,STBI__F_paeth=4,STBI__F_avg_first};static l ew[5]={
STBI__F_none,STBI__F_sub,STBI__F_none,STBI__F_avg_first,STBI__F_sub};static int
ex(int tn,int to,int tp){int tq=tp*3-(tn+to);int tr=tn<to?tn:to;int ts=tn<to?to
:tn;int tt=(ts<=tq)?tr:tp;int tu=(tq<=tr)?ts:tt;return tu;}static const l ey[9]
={0,0xff,0x55,0,0x11,0,0,0,0x01};static void ez(l*tn,l*to,uint32_t tp,int tq){
int tr;if(tq==1){for(tr=tp-1;tr>=0;--tr){tn[tr*2+1]=255;tn[tr*2+0]=to[tr];}}///
else{assert(tq==3);for(tr=tp-1;tr>=0;--tr){tn[tr*4+3]=255;tn[tr*4+2]=to[tr*3+2]
;tn[tr*4+1]=to[tr*3+1];tn[tr*4+0]=to[tr*3+0];}}}static int fa(ev*tn,l*to,//////
uint32_t tp,int tq,uint32_t tr,uint32_t ts,int tt,int tu){int tv=(tt==16?2:1);
ab*tw=tn->l;uint32_t tx,ty,tz=tr*tq*tv;uint32_t ua,ub;l*uc;int ud=1;int ue;int
uf=tw->ab;int ug=tq*tv;int uh=uf*tv;int ui=tr;assert(tq==tw->ab||tq==tw->ab+1);
tn->ac=(l*)ay(tr,ts,ug,0);if(!tn->ac)return ar("outofmem");if(!aw(uf,tr,tt,7))
return ar("too large");ub=(((uf*tr*tt)+7)>>3);if(!av(ub,ts,ub))return ar(//////
"too large");ua=(ub+1)*ts;if(tp<ua)return ar("not enough pixels");uc=(l*)ax(ub,
2,0);if(!uc)return ar("outofmem");if(tt<8){uh=1;ui=ub;}for(ty=0;ty<ts;++ty){l*
uj=uc+(ty&1)*ub;l*uk=uc+(~ty&1)*ub;l*ul=tn->ac+tz*ty;int um=ui*uh;int un=*to++;
if(un>4){ud=ar("invalid filter");break;}if(ty==0)un=ew[un];switch(un){case/////
STBI__F_none:memcpy(uj,to,um);break;case STBI__F_sub:memcpy(uj,to,uh);for(ue=uh
;ue<um;++ue)uj[ue]=((l)((to[ue]+uj[ue-uh])&255));break;case STBI__F_up:for(ue=0
;ue<um;++ue)uj[ue]=((l)((to[ue]+uk[ue])&255));break;case STBI__F_avg:for(ue=0;
ue<uh;++ue)uj[ue]=((l)((to[ue]+(uk[ue]>>1))&255));for(ue=uh;ue<um;++ue)uj[ue]=(
(l)((to[ue]+((uk[ue]+uj[ue-uh])>>1))&255));break;case STBI__F_paeth:for(ue=0;ue
<uh;++ue)uj[ue]=((l)((to[ue]+uk[ue])&255));for(ue=uh;ue<um;++ue)uj[ue]=((l)((to
[ue]+ex(uj[ue-uh],uk[ue],uk[ue-uh]))&255));break;case STBI__F_avg_first:memcpy(
uj,to,uh);for(ue=uh;ue<um;++ue)uj[ue]=((l)((to[ue]+(uj[ue-uh]>>1))&255));break;
}to+=um;if(tt<8){l uo=(tu==0)?ey[tt]:1;l*up=uj;l*uq=ul;l ur=0;uint32_t us=tr*uf
;if(tt==4){for(tx=0;tx<us;++tx){if((tx&1)==0)ur=*up++;*uq++ =uo*(ur>>4);ur<<=4;
}}else if(tt==2){for(tx=0;tx<us;++tx){if((tx&3)==0)ur=*up++;*uq++ =uo*(ur>>6);
ur<<=2;}}else{assert(tt==1);for(tx=0;tx<us;++tx){if((tx&7)==0)ur=*up++;*uq++ =
uo*(ur>>7);ur<<=1;}}if(uf!=tq)ez(ul,ul,tr,uf);}else if(tt==8){if(uf==tq)memcpy(
ul,uj,tr*uf);else ez(ul,uj,tr,uf);}else if(tt==16){uint16_t*ut=(uint16_t*)ul;//
uint32_t uu=tr*uf;if(uf==tq){for(tx=0;tx<uu;++tx,++ut,uj+=2)*ut=(uj[0]<<8)|uj[1
];}else{assert(uf+1==tq);if(uf==1){for(tx=0;tx<tr;++tx,ut+=2,uj+=2){ut[0]=(uj[0
]<<8)|uj[1];ut[1]=0xffff;}}else{assert(uf==3);for(tx=0;tx<tr;++tx,ut+=4,uj+=6){
ut[0]=(uj[0]<<8)|uj[1];ut[1]=(uj[2]<<8)|uj[3];ut[2]=(uj[4]<<8)|uj[5];ut[3]=////
0xffff;}}}}}free(uc);if(!ud)return 0;return 1;}static int fb(ev*tn,l*to,///////
uint32_t tp,int tq,int tr,int ts,int tt){int tu=(tr==16?2:1);int tv=tq*tu;l*tw;
int tx;if(!tt)return fa(tn,to,tp,tq,tn->l->l,tn->l->u,tr,ts);tw=(l*)ay(tn->l->l
,tn->l->u,tv,0);if(!tw)return ar("outofmem");for(tx=0;tx<7;++tx){int ty[]={0,4,
0,2,0,1,0};int tz[]={0,0,4,0,2,0,1};int ua[]={8,8,4,4,2,2,1};int ub[]={8,8,8,4,
4,2,2};int uc,ud,ue,uf;ue=(tn->l->l-ty[tx]+ua[tx]-1)/ua[tx];uf=(tn->l->u-tz[tx]
+ub[tx]-1)/ub[tx];if(ue&&uf){uint32_t ug=((((tn->l->ab*ue*tr)+7)>>3)+1)*uf;if(!
fa(tn,to,tp,tq,ue,uf,tr,ts)){free(tw);return 0;}for(ud=0;ud<uf;++ud){for(uc=0;
uc<ue;++uc){int uh=ud*ub[tx]+tz[tx];int ui=uc*ua[tx]+ty[tx];memcpy(tw+uh*tn->l
->l*tv+ui*tv,tn->ac+(ud*ue+uc)*tv,tv);}}free(tn->ac);to+=ug;tp-=ug;}}tn->ac=tw;
return 1;}static int fc(ev*tn,l to[3],int tp){ab*tq=tn->l;uint32_t tr,ts=tq->l*
tq->u;l*tt=tn->ac;assert(tp==2||tp==4);if(tp==2){for(tr=0;tr<ts;++tr){tt[1]=(tt
[0]==to[0]?0:255);tt+=2;}}else{for(tr=0;tr<ts;++tr){if(tt[0]==to[0]&&tt[1]==to[
1]&&tt[2]==to[2])tt[3]=0;tt+=4;}}return 1;}static int fd(ev*tn,uint16_t to[3],
int tp){ab*tq=tn->l;uint32_t tr,ts=tq->l*tq->u;uint16_t*tt=(uint16_t*)tn->ac;//
assert(tp==2||tp==4);if(tp==2){for(tr=0;tr<ts;++tr){tt[1]=(tt[0]==to[0]?0:65535
);tt+=2;}}else{for(tr=0;tr<ts;++tr){if(tt[0]==to[0]&&tt[1]==to[1]&&tt[2]==to[2]
)tt[3]=0;tt+=4;}}return 1;}static int fe(ev*tn,l*to,int tp,int tq){uint32_t tr,
ts=tn->l->l*tn->l->u;l*tt,*tu,*tv=tn->ac;tt=(l*)ax(ts,tq,0);if(tt==NULL)return
ar("outofmem");tu=tt;if(tq==3){for(tr=0;tr<ts;++tr){int tw=tv[tr]*4;tt[0]=to[tw
];tt[1]=to[tw+1];tt[2]=to[tw+2];tt+=3;}}else{for(tr=0;tr<ts;++tr){int tx=tv[tr]
*4;tt[0]=to[tx];tt[1]=to[tx+1];tt[2]=to[tx+2];tt[3]=to[tx+3];tt+=4;}}free(tn->
ac);tn->ac=tu;(void)sizeof(tp);return 1;}static int ff=0;static int fg=0;static
_Thread_local int fh,fi;static _Thread_local int fj,fk;static void fl(ev*tn){ab
*to=tn->l;uint32_t tp,tq=to->l*to->u;l*tr=tn->ac;if(to->ac==3){for(tp=0;tp<tq;
++tp){l ts=tr[0];tr[0]=tr[2];tr[2]=ts;tr+=3;}}else{assert(to->ac==4);if((fi?fh:
ff)){for(tp=0;tp<tq;++tp){l tt=tr[3];l tu=tr[0];if(tt){l tv=tt/2;tr[0]=(tr[2]*
255+tv)/tt;tr[1]=(tr[1]*255+tv)/tt;tr[2]=(tu*255+tv)/tt;}else{tr[0]=tr[2];tr[2]
=tu;}tr+=4;}}else{for(tp=0;tp<tq;++tp){l tw=tr[0];tr[0]=tr[2];tr[2]=tw;tr+=4;}}
}}static int fm(ev*tn,int to,int tp){l tq[1024],tr=0;l ts=0,tt[3]={0};uint16_t
tu[3];uint32_t tv=0,tw=0,tx,ty=0;int tz=1,ua,ub=0,uc=0,ud=0;ab*ue=tn->l;tn->ab=
NULL;tn->u=NULL;tn->ac=NULL;if(!eu(ue))return 0;if(to==STBI__SCAN_type)return 1
;for(;;){es uf=et(ue);switch(uf.u){case(((unsigned)('C')<<24)+((unsigned)('g')
<<16)+((unsigned)('B')<<8)+(unsigned)('I')):ud=1;bm(ue,uf.l);break;case(((/////
unsigned)('I')<<24)+((unsigned)('H')<<16)+((unsigned)('D')<<8)+(unsigned)('R'))
:{int ug,uh;if(!tz)return ar("multiple IHDR");tz=0;if(uf.l!=13)return ar(//////
"bad IHDR len");ue->l=bp(ue);ue->u=bp(ue);if(ue->u>(1<<24))return ar(//////////
"too large");if(ue->l>(1<<24))return ar("too large");tn->ad=bk(ue);if(tn->ad!=1
&&tn->ad!=2&&tn->ad!=4&&tn->ad!=8&&tn->ad!=16)return ar("1/2/4/8/16-bit only");
uc=bk(ue);if(uc>6)return ar("bad ctype");if(uc==3&&tn->ad==16)return ar(///////
"bad ctype");if(uc==3)tr=3;else if(uc&1)return ar("bad ctype");ug=bk(ue);if(ug)
return ar("bad comp method");uh=bk(ue);if(uh)return ar("bad filter method");ub=
bk(ue);if(ub>1)return ar("bad interlace method");if(!ue->l||!ue->u)return ar(//
"0-pixel image");if(!tr){ue->ab=(uc&2?3:1)+(uc&4?1:0);if((1<<30)/ue->l/ue->ab<
ue->u)return ar("too large");}else{ue->ab=1;if((1<<30)/ue->l/4<ue->u)return ar(
"too large");}break;}case(((unsigned)('P')<<24)+((unsigned)('L')<<16)+((///////
unsigned)('T')<<8)+(unsigned)('E')):{if(tz)return ar("first not IHDR");if(uf.l>
256*3)return ar("invalid PLTE");ty=uf.l/3;if(ty*3!=uf.l)return ar(/////////////
"invalid PLTE");for(tx=0;tx<ty;++tx){tq[tx*4+0]=bk(ue);tq[tx*4+1]=bk(ue);tq[tx*
4+2]=bk(ue);tq[tx*4+3]=255;}break;}case(((unsigned)('t')<<24)+((unsigned)('R')
<<16)+((unsigned)('N')<<8)+(unsigned)('S')):{if(tz)return ar("first not IHDR");
if(tn->u)return ar("tRNS after IDAT");if(tr){if(to==STBI__SCAN_header){ue->ab=4
;return 1;}if(ty==0)return ar("tRNS before PLTE");if(uf.l>ty)return ar(////////
"bad tRNS len");tr=4;for(tx=0;tx<uf.l;++tx)tq[tx*4+3]=bk(ue);}else{if(!(ue->ab&
1))return ar("tRNS with alpha");if(uf.l!=(uint32_t)ue->ab*2)return ar(/////////
"bad tRNS len");ts=1;if(to==STBI__SCAN_header){++ue->ab;return 1;}if(tn->ad==16
){for(ua=0;ua<ue->ab&&ua<3;++ua)tu[ua]=(uint16_t)bo(ue);}else{for(ua=0;ua<ue->
ab&&ua<3;++ua)tt[ua]=(l)(bo(ue)&255)*ey[tn->ad];}}break;}case(((unsigned)('I')
<<24)+((unsigned)('D')<<16)+((unsigned)('A')<<8)+(unsigned)('T')):{if(tz)return
ar("first not IHDR");if(tr&&!ty)return ar("no PLTE");if(to==STBI__SCAN_header){
if(tr)ue->ab=tr;return 1;}if(uf.l>(1u<<30))return ar("IDAT size limit");if((int
)(tv+uf.l)<(int)tv)return 0;if(tv+uf.l>tw){uint32_t ui=tw;l*uj;if(tw==0)tw=uf.l
>4096?uf.l:4096;while(tv+uf.l>tw)tw*=2;(void)sizeof(ui);uj=(l*)realloc(tn->u,tw
);if(uj==NULL)return ar("outofmem");tn->u=uj;}if(!bn(ue,tn->u+tv,uf.l))return//
ar("outofdata");tv+=uf.l;break;}case(((unsigned)('I')<<24)+((unsigned)('E')<<16
)+((unsigned)('N')<<8)+(unsigned)('D')):{uint32_t uk,ul;if(tz)return ar(///////
"first not IHDR");if(to!=STBI__SCAN_load)return 1;if(tn->u==NULL)return ar(////
"no IDAT");ul=(ue->l*tn->ad+7)/8;uk=ul*ue->u*ue->ab+ue->u;tn->ab=(l*)//////////
stbi_zlib_decode_malloc_guesssize_headerflag((char*)tn->u,tv,uk,(int*)&uk,!ud);
if(tn->ab==NULL)return 0;free(tn->u);tn->u=NULL;if((tp==ue->ab+1&&tp!=3&&!tr)||
ts)ue->ac=ue->ab+1;else ue->ac=ue->ab;if(!fb(tn,tn->ab,uk,ue->ac,tn->ad,uc,ub))
return 0;if(ts){if(tn->ad==16){if(!fd(tn,tu,ue->ac))return 0;}else{if(!fc(tn,tt
,ue->ac))return 0;}}if(ud&&(fk?fj:fg)&&ue->ac>2)fl(tn);if(tr){ue->ab=tr;ue->ac=
tr;if(tp>=3)ue->ac=tp;if(!fe(tn,tq,ty,ue->ac))return 0;}else if(ts){++ue->ab;}
free(tn->ab);tn->ab=NULL;bp(ue);return 1;}default:if(tz)return ar(/////////////
"first not IHDR");if((uf.u&(1<<29))==0){static char um[]=//////////////////////
"XXXX PNG chunk not known";um[0]=((l)((uf.u>>24)&255));um[1]=((l)((uf.u>>16)&//
255));um[2]=((l)((uf.u>>8)&255));um[3]=((l)((uf.u>>0)&255));return ar(um);}bm(
ue,uf.l);break;}bp(ue);}}static void*fn(ev*tn,int*to,int*tp,int*tq,int tr,al*ts
){void*tt=NULL;if(tr<0||tr>4)return((unsigned char*)(size_t)(ar("bad req_comp")
?NULL:NULL));if(fm(tn,STBI__SCAN_load,tr)){if(tn->ad<=8)ts->l=8;else if(tn->ad
==16)ts->l=16;else return((unsigned char*)(size_t)(ar("bad bits_per_channel")?
NULL:NULL));tt=tn->ac;tn->ac=NULL;if(tr&&tr!=tn->l->ac){if(ts->l==8)tt=br((////
unsigned char*)tt,tn->l->ac,tr,tn->l->l,tn->l->u);else tt=bt((uint16_t*)tt,tn->
l->ac,tr,tn->l->l,tn->l->u);tn->l->ac=tr;if(tt==NULL)return tt;}*to=tn->l->l;*
tp=tn->l->u;if(tq)*tq=tn->l->ab;}free(tn->ac);tn->ac=NULL;free(tn->ab);tn->ab=
NULL;free(tn->u);tn->u=NULL;return tt;}static void*ap(ab*tt,int*tu,int*tv,int*
tw,int tx,al*ty){ev tz;tz.l=tt;return fn(&tz,tu,tv,tw,tx,ty);}static int ao(ab*
to){int tp;tp=eu(to);ak(to);return tp;}typedef uint8_t fo;typedef uint32_t fp;
typedef uint64_t fq;typedef enum{STBIR_1CHANNEL=1,STBIR_2CHANNEL=2,STBIR_RGB=3,
STBIR_BGR=0,STBIR_4CHANNEL=5,STBIR_RGBA=4,STBIR_BGRA=6,STBIR_ARGB=7,STBIR_ABGR=
8,STBIR_RA=9,STBIR_AR=10,STBIR_RGBA_PM=11,STBIR_BGRA_PM=12,STBIR_ARGB_PM=13,///
STBIR_ABGR_PM=14,STBIR_RA_PM=15,STBIR_AR_PM=16,STBIR_RGBA_NO_AW=11,////////////
STBIR_BGRA_NO_AW=12,STBIR_ARGB_NO_AW=13,STBIR_ABGR_NO_AW=14,STBIR_RA_NO_AW=15,
STBIR_AR_NO_AW=16,}fr;typedef enum{STBIR_EDGE_CLAMP=0,STBIR_EDGE_REFLECT=1,////
STBIR_EDGE_WRAP=2,STBIR_EDGE_ZERO=3,}fs;typedef enum{STBIR_FILTER_DEFAULT=0,///
STBIR_FILTER_BOX=1,STBIR_FILTER_TRIANGLE=2,STBIR_FILTER_CUBICBSPLINE=3,////////
STBIR_FILTER_CATMULLROM=4,STBIR_FILTER_MITCHELL=5,STBIR_FILTER_POINT_SAMPLE=6,
STBIR_FILTER_OTHER=7,}ft;typedef enum{STBIR_TYPE_UINT8=0,STBIR_TYPE_UINT8_SRGB=
1,STBIR_TYPE_UINT8_SRGB_ALPHA=2,STBIR_TYPE_UINT16=3,STBIR_TYPE_FLOAT=4,////////
STBIR_TYPE_HALF_FLOAT=5}fu;extern void*stbir_resize(const void*tn,int to,int tp
,int tq,void*tr,int ts,int tt,int tu,fr tv,fu tw,fs tx,ft ty);typedef void/////
const*fv(void*optional_output,void const*input_ptr,int num_pixels,int x,int y,
void*context);typedef void fw(void const*output_ptr,int num_pixels,int y,void*
context);typedef float fx(float x,float scale,void*user_data);typedef float fy(
float scale,void*user_data);typedef struct stbir__info fz;typedef struct///////
STBIR_RESIZE{void*l;void const*u;int ab,ac;double ad,ae,ag,ah;fv*ai;void*aj;int
ak,al;int am,an,ao,ap;fw*aq;int ar;int as;int at;int au;int av;int aw;fr ax;fr
ay;fu az;fu ba;ft bb,bc;fs bd,be;fx*bg;fy*bh;fx*bi;fy*bj;fz*bk;}ga;extern void
stbir_resize_init(ga*tn,const void*to,int tp,int tq,int tr,void*ts,int tt,int//
tu,int tv,fr tw,fu tx);extern int stbir_build_samplers(ga*tn);extern void//////
stbir_free_samplers(ga*tn);extern int stbir_resize_extended(ga*tn);extern int//
stbir_build_samplers_with_splits(ga*tn,int to);typedef enum{STBIRI_1CHANNEL=0,
STBIRI_2CHANNEL=1,STBIRI_RGB=2,STBIRI_BGR=3,STBIRI_4CHANNEL=4,STBIRI_RGBA=5,///
STBIRI_BGRA=6,STBIRI_ARGB=7,STBIRI_ABGR=8,STBIRI_RA=9,STBIRI_AR=10,////////////
STBIRI_RGBA_PM=11,STBIRI_BGRA_PM=12,STBIRI_ARGB_PM=13,STBIRI_ABGR_PM=14,///////
STBIRI_RA_PM=15,STBIRI_AR_PM=16,}gb;static unsigned char gd[]={1,1,1,2,4,2};///
typedef struct{int l;int u;}ge;typedef struct{int l;int u;int ab;}gf;typedef///
struct{int l;int u;int ab;}gg;typedef struct stbir__scale_info{int l;int u;////
float ab;float ac;float ad;int ae;fp ag,ah;}gh;typedef struct{ge*l;float*u;ge*
ab;float*ac;gh ad;float ae;ft ag;fx*ah;fy*ai;fs aj;int ak;int al;int am;int an;
int ao;int ap;gf aq;int ar;int as;int at;int au;int av;}gi;typedef struct{ge l;
int u[2];gg ab[2];}gj;typedef struct{float*l;int u;int ab;int ac;int ad,ae;int
ag,ah;float*ai;float*aj;char ak[64];}gk;typedef float*gl(float*decode,int//////
width_times_channels,void const*input);typedef void gm(float*decode_buffer,int
width_times_channels);typedef void gn(float*output_buffer,unsigned int/////////
output_sub_size,float const*decode_buffer,ge const*horizontal_contributors,////
float const*horizontal_coefficients,int coefficient_width);typedef void go(////
float*encode_buffer,int width_times_channels);typedef void gp(void*output,int//
width_times_channels,float const*encode);struct stbir__info{gi l;gi u;void/////
const*ab;void*ac;int ad;int ae;int ag;int ah;fu ai;fu aj;fv*ak;void*al;fw*am;gj
an;void*ao;gk*ap;gl*aq;gm*ar;gn*as;go*at;gp*au;int av;int aw;gb ax;gb ay;int az
;int ba,bb;int bc;int bd;int be;size_t bg;};static __inline__ int gq(int tn,int
to){return tn<to?tn:to;}static __inline__ int gr(int tn,int to){return tn>to?tn
:to;}static float gs[256]={0.000000f,0.000304f,0.000607f,0.000911f,0.001214f,//
0.001518f,0.001821f,0.002125f,0.002428f,0.002732f,0.003035f,0.003347f,0.003677f
,0.004025f,0.004391f,0.004777f,0.005182f,0.005605f,0.006049f,0.006512f,////////
0.006995f,0.007499f,0.008023f,0.008568f,0.009134f,0.009721f,0.010330f,0.010960f
,0.011612f,0.012286f,0.012983f,0.013702f,0.014444f,0.015209f,0.015996f,////////
0.016807f,0.017642f,0.018500f,0.019382f,0.020289f,0.021219f,0.022174f,0.023153f
,0.024158f,0.025187f,0.026241f,0.027321f,0.028426f,0.029557f,0.030713f,////////
0.031896f,0.033105f,0.034340f,0.035601f,0.036889f,0.038204f,0.039546f,0.040915f
,0.042311f,0.043735f,0.045186f,0.046665f,0.048172f,0.049707f,0.051269f,////////
0.052861f,0.054480f,0.056128f,0.057805f,0.059511f,0.061246f,0.063010f,0.064803f
,0.066626f,0.068478f,0.070360f,0.072272f,0.074214f,0.076185f,0.078187f,////////
0.080220f,0.082283f,0.084376f,0.086500f,0.088656f,0.090842f,0.093059f,0.095307f
,0.097587f,0.099899f,0.102242f,0.104616f,0.107023f,0.109462f,0.111932f,////////
0.114435f,0.116971f,0.119538f,0.122139f,0.124772f,0.127438f,0.130136f,0.132868f
,0.135633f,0.138432f,0.141263f,0.144128f,0.147027f,0.149960f,0.152926f,////////
0.155926f,0.158961f,0.162029f,0.165132f,0.168269f,0.171441f,0.174647f,0.177888f
,0.181164f,0.184475f,0.187821f,0.191202f,0.194618f,0.198069f,0.201556f,////////
0.205079f,0.208637f,0.212231f,0.215861f,0.219526f,0.223228f,0.226966f,0.230740f
,0.234551f,0.238398f,0.242281f,0.246201f,0.250158f,0.254152f,0.258183f,////////
0.262251f,0.266356f,0.270498f,0.274677f,0.278894f,0.283149f,0.287441f,0.291771f
,0.296138f,0.300544f,0.304987f,0.309469f,0.313989f,0.318547f,0.323143f,////////
0.327778f,0.332452f,0.337164f,0.341914f,0.346704f,0.351533f,0.356400f,0.361307f
,0.366253f,0.371238f,0.376262f,0.381326f,0.386430f,0.391573f,0.396755f,////////
0.401978f,0.407240f,0.412543f,0.417885f,0.423268f,0.428691f,0.434154f,0.439657f
,0.445201f,0.450786f,0.456411f,0.462077f,0.467784f,0.473532f,0.479320f,////////
0.485150f,0.491021f,0.496933f,0.502887f,0.508881f,0.514918f,0.520996f,0.527115f
,0.533276f,0.539480f,0.545725f,0.552011f,0.558340f,0.564712f,0.571125f,////////
0.577581f,0.584078f,0.590619f,0.597202f,0.603827f,0.610496f,0.617207f,0.623960f
,0.630757f,0.637597f,0.644480f,0.651406f,0.658375f,0.665387f,0.672443f,////////
0.679543f,0.686685f,0.693872f,0.701102f,0.708376f,0.715694f,0.723055f,0.730461f
,0.737911f,0.745404f,0.752942f,0.760525f,0.768151f,0.775822f,0.783538f,////////
0.791298f,0.799103f,0.806952f,0.814847f,0.822786f,0.830770f,0.838799f,0.846873f
,0.854993f,0.863157f,0.871367f,0.879622f,0.887923f,0.896269f,0.904661f,////////
0.913099f,0.921582f,0.930111f,0.938686f,0.947307f,0.955974f,0.964686f,0.973445f
,0.982251f,0.991102f,1.0f};typedef union{unsigned int l;float u;}gt;static/////
const fp gu[104]={0x0073000d,0x007a000d,0x0080000d,0x0087000d,0x008d000d,//////
0x0094000d,0x009a000d,0x00a1000d,0x00a7001a,0x00b4001a,0x00c1001a,0x00ce001a,//
0x00da001a,0x00e7001a,0x00f4001a,0x0101001a,0x010e0033,0x01280033,0x01410033,//
0x015b0033,0x01750033,0x018f0033,0x01a80033,0x01c20033,0x01dc0067,0x020f0067,//
0x02430067,0x02760067,0x02aa0067,0x02dd0067,0x03110067,0x03440067,0x037800ce,//
0x03df00ce,0x044600ce,0x04ad00ce,0x051400ce,0x057b00c5,0x05dd00bc,0x063b00b5,//
0x06970158,0x07420142,0x07e30130,0x087b0120,0x090b0112,0x09940106,0x0a1700fc,//
0x0a9500f2,0x0b0f01cb,0x0bf401ae,0x0ccb0195,0x0d950180,0x0e56016e,0x0f0d015e,//
0x0fbc0150,0x10630143,0x11070264,0x1238023e,0x1357021d,0x14660201,0x156601e9,//
0x165a01d3,0x174401c0,0x182401af,0x18fe0331,0x1a9602fe,0x1c1502d2,0x1d7e02ad,//
0x1ed4028d,0x201a0270,0x21520256,0x227d0240,0x239f0443,0x25c003fe,0x27bf03c4,//
0x29a10392,0x2b6a0367,0x2d1d0341,0x2ebe031f,0x304d0300,0x31d105b0,0x34a80555,//
0x37520507,0x39d504c5,0x3c37048b,0x3e7c0458,0x40a8042a,0x42bd0401,0x44c20798,//
0x488e071e,0x4c1c06b6,0x4f76065d,0x52a50610,0x55ac05cc,0x5892058f,0x5b590559,//
0x5e0c0a23,0x631c0980,0x67db08f6,0x6c55087f,0x70940818,0x74a007bd,0x787d076c,//
0x7c330723,};static __inline__ fo gv(float tn){static const gt to={0x3f7fffff};
static const gt tp={(127-13)<<23};fp tq,tr,ts,tt;gt tu;if(!(tn>tp.u))return 0;
if(tn>to.u)return 255;tu.u=tn;tq=gu[(tu.l-tp.l)>>20];tr=(tq>>16)<<9;ts=tq&/////
0xffff;tt=(tu.l>>12)&0xff;return(unsigned char)((tr+ts*tt)>>16);}static const//
__m128 gx={0.0f,1.0f,0.0f,1.0f};static __m128i gy={(long long)((((fq)(fp)(32768
))<<32)|((fq)(fp)(32768))),(long long)((((fq)(fp)(32768))<<32)|((fq)(fp)(32768)
))};static __m128i gz={(long long)((((fq)(fp)(((32768<<16)|32768)))<<32)|((fq)(
fp)(((32768<<16)|32768)))),(long long)((((fq)(fp)(((32768<<16)|32768)))<<32)|((
fq)(fp)(((32768<<16)|32768))))};static __inline__ float hb(float tn){__m128 to=
_mm_set_ss(tn);__m128 tp=_mm_cvtepi32_ps(_mm_cvttps_epi32(to));__m128 tq=//////
_mm_add_ss(tp,_mm_and_ps(_mm_cmplt_ss(to,tp),_mm_set_ss(-1.0f)));return////////
_mm_cvtss_f32(tq);}static __inline__ float hd(float tn){__m128 to=_mm_set_ss(tn
);__m128 tp=_mm_cvtepi32_ps(_mm_cvttps_epi32(to));__m128 tq=_mm_add_ss(tp,/////
_mm_and_ps(_mm_cmplt_ss(tp,to),_mm_set_ss(1.0f)));return _mm_cvtss_f32(tq);}///
typedef union stbir__FP16{unsigned short l;}he;static __inline__ float hf(he tn
){static const gt to={(254-15)<<23};static const gt tp={(127+16)<<23};gt tq;tq.
l=(tn.l&0x7fff)<<13;tq.u*=to.u;if(tq.u>=tp.u)tq.l|=255<<23;tq.l|=(tn.l&0x8000)
<<16;return tq.u;}static __inline__ he hg(float tn){gt to={255<<23};gt tp={(127
+16)<<23};gt tq={((127-15)+(23-10)+1)<<23};unsigned int tr=0x80000000u;he ts={0
};gt tt;unsigned int tu;tt.u=tn;tu=tt.l&tr;tt.l^=tu;if(tt.l>=tp.l)ts.l=(tt.l>to
.l)?0x7e00:0x7c00;else{if(tt.l<(113<<23)){tt.u+=tq.u;ts.l=(unsigned short)(tt.l
-tq.l);}else{unsigned int tv=(tt.l>>13)&1;tt.l=tt.l+((15u-127)<<23)+0xfff;tt.l
+=tv;ts.l=(unsigned short)(tt.l>>13);}}ts.l|=tu>>16;return ts;}__inline__//////
static void hh(float*tn,void const*to){static const __m128i tp={(long long)((((
fq)(fp)(0x7fff))<<32)|((fq)(fp)(0x7fff))),(long long)((((fq)(fp)(0x7fff))<<32)|
((fq)(fp)(0x7fff)))};static const __m128i tq={(long long)((((fq)(fp)(0x0400))<<
32)|((fq)(fp)(0x0400))),(long long)((((fq)(fp)(0x0400))<<32)|((fq)(fp)(0x0400))
)};static const __m128i tr={(long long)((((fq)(fp)(0x7c00))<<32)|((fq)(fp)(////
0x7c00))),(long long)((((fq)(fp)(0x7c00))<<32)|((fq)(fp)(0x7c00)))};static/////
const __m128i ts={(long long)((((fq)(fp)((127-15)<<23))<<32)|((fq)(fp)((127-15)
<<23))),(long long)((((fq)(fp)((127-15)<<23))<<32)|((fq)(fp)((127-15)<<23)))};
static const __m128i tt={(long long)((((fq)(fp)(113<<23))<<32)|((fq)(fp)(113<<
23))),(long long)((((fq)(fp)(113<<23))<<32)|((fq)(fp)(113<<23)))};__m128i tu=//
_mm_loadu_si128((__m128i const*)(to));__m128i tv=_mm_unpacklo_epi16(tu,////////
_mm_setzero_si128());__m128i tw=(tp);__m128i tx=(ts);__m128i ty=(tq);__m128i tz
=(tr);__m128i ua=_mm_and_si128(tw,tv);__m128i ub=_mm_xor_si128(tv,ua);__m128i//
uc=_mm_cmpgt_epi32(tz,ua);__m128i ud=_mm_cmpgt_epi32(ty,ua);__m128i ue=////////
_mm_slli_epi32(ua,13);__m128i uf=_mm_andnot_si128(uc,tx);__m128i ug=///////////
_mm_add_epi32(tx,ue);__m128i uh=_mm_add_epi32(ue,(tt));__m128i ui=_mm_add_epi32
(ug,uf);__m128 uj=_mm_sub_ps(_mm_castsi128_ps(uh),*(const __m128*)&tt);__m128//
uk=_mm_and_ps(uj,_mm_castsi128_ps(ud));__m128 ul=_mm_andnot_ps(_mm_castsi128_ps
(ud),_mm_castsi128_ps(ui));__m128 um=_mm_or_ps(uk,ul);__m128i un=_mm_slli_epi32
(ub,16);__m128 uo=_mm_or_ps(um,_mm_castsi128_ps(un));_mm_storeu_ps((float*)(tn+
0),uo);tv=_mm_unpackhi_epi16(tu,_mm_setzero_si128());ua=_mm_and_si128(tw,tv);ub
=_mm_xor_si128(tv,ua);uc=_mm_cmpgt_epi32(tz,ua);ud=_mm_cmpgt_epi32(ty,ua);ue=//
_mm_slli_epi32(ua,13);uf=_mm_andnot_si128(uc,tx);ug=_mm_add_epi32(tx,ue);uh=///
_mm_add_epi32(ue,(tt));ui=_mm_add_epi32(ug,uf);uj=_mm_sub_ps(_mm_castsi128_ps(
uh),*(const __m128*)&tt);uk=_mm_and_ps(uj,_mm_castsi128_ps(ud));ul=////////////
_mm_andnot_ps(_mm_castsi128_ps(ud),_mm_castsi128_ps(ui));um=_mm_or_ps(uk,ul);un
=_mm_slli_epi32(ub,16);uo=_mm_or_ps(um,_mm_castsi128_ps(un));_mm_storeu_ps((///
float*)(tn+4),uo);}__inline__ static void hj(void*tn,float const*to){static////
const __m128i tp={(long long)((((fq)(fp)(0x80000000u))<<32)|((fq)(fp)(/////////
0x80000000u))),(long long)((((fq)(fp)(0x80000000u))<<32)|((fq)(fp)(0x80000000u)
))};static const __m128i tq={(long long)((((fq)(fp)((127+16)<<23))<<32)|((fq)(
fp)((127+16)<<23))),(long long)((((fq)(fp)((127+16)<<23))<<32)|((fq)(fp)((127+
16)<<23)))};static const __m128i tr={(long long)((((fq)(fp)(0x200))<<32)|((fq)(
fp)(0x200))),(long long)((((fq)(fp)(0x200))<<32)|((fq)(fp)(0x200)))};static////
const __m128i ts={(long long)((((fq)(fp)(0x7c00))<<32)|((fq)(fp)(0x7c00))),(///
long long)((((fq)(fp)(0x7c00))<<32)|((fq)(fp)(0x7c00)))};static const __m128i//
tt={(long long)((((fq)(fp)((127-14)<<23))<<32)|((fq)(fp)((127-14)<<23))),(long
long)((((fq)(fp)((127-14)<<23))<<32)|((fq)(fp)((127-14)<<23)))};static const///
__m128i tu={(long long)((((fq)(fp)(((127-15)+(23-10)+1)<<23))<<32)|((fq)(fp)(((
127-15)+(23-10)+1)<<23))),(long long)((((fq)(fp)(((127-15)+(23-10)+1)<<23))<<32
)|((fq)(fp)(((127-15)+(23-10)+1)<<23)))};static const __m128i tv={(long long)((
((fq)(fp)(0xfff-((127-15)<<23)))<<32)|((fq)(fp)(0xfff-((127-15)<<23)))),(long//
long)((((fq)(fp)(0xfff-((127-15)<<23)))<<32)|((fq)(fp)(0xfff-((127-15)<<23))))}
;__m128 tw=_mm_loadu_ps(to);__m128 tx=_mm_castsi128_ps((tp));__m128 ty=////////
_mm_and_ps(tx,tw);__m128 tz=_mm_xor_ps(tw,ty);__m128i ua=_mm_castps_si128(tz);
__m128i ub=(tq);__m128 uc=_mm_cmpunord_ps(tz,tz);__m128i ud=_mm_cmpgt_epi32(ub,
ua);__m128i ue=_mm_and_si128(_mm_castps_si128(uc),(tr));__m128i uf=_mm_or_si128
(ue,(ts));__m128i ug=(tt);__m128i uh=_mm_cmpgt_epi32(ug,ua);__m128 ui=/////////
_mm_add_ps(tz,_mm_castsi128_ps((tu)));__m128i uj=_mm_sub_epi32(_mm_castps_si128
(ui),(tu));__m128i uk=_mm_slli_epi32(ua,31-13);__m128i ul=_mm_srai_epi32(uk,31)
;__m128i um=_mm_add_epi32(ua,(tv));__m128i un=_mm_sub_epi32(um,ul);__m128i uo=
_mm_srli_epi32(un,13);__m128i up=_mm_or_si128(_mm_and_si128(uj,uh),////////////
_mm_andnot_si128(uh,uo));__m128i uq=_mm_or_si128(_mm_and_si128(up,ud),/////////
_mm_andnot_si128(ud,uf));__m128i ur=_mm_srai_epi32(_mm_castps_si128(ty),16);///
__m128i us,ut=_mm_or_si128(uq,ur);tw=_mm_loadu_ps(to+4);ty=_mm_and_ps(tx,tw);tz
=_mm_xor_ps(tw,ty);ua=_mm_castps_si128(tz);uc=_mm_cmpunord_ps(tz,tz);ud=///////
_mm_cmpgt_epi32(ub,ua);ue=_mm_and_si128(_mm_castps_si128(uc),tr);uf=///////////
_mm_or_si128(ue,(ts));uh=_mm_cmpgt_epi32(ug,ua);ui=_mm_add_ps(tz,//////////////
_mm_castsi128_ps((tu)));uj=_mm_sub_epi32(_mm_castps_si128(ui),(tu));uk=////////
_mm_slli_epi32(ua,31-13);ul=_mm_srai_epi32(uk,31);um=_mm_add_epi32(ua,(tv));un=
_mm_sub_epi32(um,ul);uo=_mm_srli_epi32(un,13);up=_mm_or_si128(_mm_and_si128(uj,
uh),_mm_andnot_si128(uh,uo));uq=_mm_or_si128(_mm_and_si128(up,ud),/////////////
_mm_andnot_si128(ud,uf));ur=_mm_srai_epi32(_mm_castps_si128(ty),16);us=////////
_mm_or_si128(uq,ur);ut=_mm_packs_epi32(ut,us);_mm_storeu_si128((__m128i*)(tn),
ut);}typedef union stbir__simdi_u32{fp l[4];int u[4];__m128i ab;}hk;static/////
const int hl[9]={0,0,0,-1,-1,-1,0,0,0};static const __m128 hm={255.0f,255.0f,//
255.0f,255.0f};static const __m128 hn={65535.0f,65535.0f,65535.0f,65535.0f};///
static const __m128 ho={3.9215689e-03f,3.9215689e-03f,3.9215689e-03f,//////////
3.9215689e-03f};static const __m128 hp={1.5259022e-05f,1.5259022e-05f,/////////
1.5259022e-05f,1.5259022e-05f};static const __m128 hq={0.5f,0.5f,0.5f,0.5f};///
static const __m128 hr={1.0f,1.0f,1.0f,1.0f};static const __m128i ht={(long////
long)((((fq)(fp)((127-13)<<23))<<32)|((fq)(fp)((127-13)<<23))),(long long)((((
fq)(fp)((127-13)<<23))<<32)|((fq)(fp)((127-13)<<23)))};static const __m128i hu=
{(long long)((((fq)(fp)(0x3f7fffff))<<32)|((fq)(fp)(0x3f7fffff))),(long long)((
((fq)(fp)(0x3f7fffff))<<32)|((fq)(fp)(0x3f7fffff)))};static const __m128i hv={(
long long)((((fq)(fp)(0xff))<<32)|((fq)(fp)(0xff))),(long long)((((fq)(fp)(0xff
))<<32)|((fq)(fp)(0xff)))};static const __m128i hw={(long long)((((fq)(fp)(////
0x02000000))<<32)|((fq)(fp)(0x02000000))),(long long)((((fq)(fp)(0x02000000))<<
32)|((fq)(fp)(0x02000000)))};static void hx(void*tn,void const*to,size_t tp){//
char*__restrict__ tq=(char*)tn;char*__restrict__ tr=((char*)tn)+tp;ptrdiff_t ts
=(char*)to-(char*)tn;assert(((tq>=((char*)to)+tp))||((tq+tp)<=(char*)to));if(tp
<(16*4)){if(tp<16){if(tp){_Pragma("GCC unroll 1")_Pragma("GCC novector")do{////
__asm__(""::"r"(tq));tq[0]=tq[ts];++tq;}while(tq<tr);}}else{__m128 tt;(tt)=////
_mm_loadu_ps((float const*)((tq+ts)));_mm_storeu_ps((float*)(tq),tt);tq=(char*)
((((size_t)tq)+16)&~15);for(;;){__asm__(""::"r"(tq));if(tq>(tr-16)){if(tq==tr)
return;tq=tr-16;}(tt)=_mm_loadu_ps((float const*)((tq+ts)));_mm_storeu_ps((////
float*)(tq),tt);tq+=16;}}}else{__m128 tu,tv,tw,tx;(tu)=_mm_loadu_ps((float/////
const*)((tq+ts)+0*4));(tv)=_mm_loadu_ps((float const*)((tq+ts)+4*4));(tw)=/////
_mm_loadu_ps((float const*)((tq+ts)+8*4));(tx)=_mm_loadu_ps((float const*)((tq+
ts)+12*4));_mm_storeu_ps((float*)(tq+0*4),tu);_mm_storeu_ps((float*)(tq+4*4),tv
);_mm_storeu_ps((float*)(tq+8*4),tw);_mm_storeu_ps((float*)(tq+12*4),tx);tq=(//
char*)((((size_t)tq)+(16*4))&~((16*4)-1));for(;;){__asm__(""::"r"(tq));if(tq>(
tr-(16*4))){if(tq==tr)return;tq=tr-(16*4);}(tu)=_mm_loadu_ps((float const*)((tq
+ts)+0*4));(tv)=_mm_loadu_ps((float const*)((tq+ts)+4*4));(tw)=_mm_loadu_ps((//
float const*)((tq+ts)+8*4));(tx)=_mm_loadu_ps((float const*)((tq+ts)+12*4));///
_mm_storeu_ps((float*)(tq+0*4),tu);_mm_storeu_ps((float*)(tq+4*4),tv);/////////
_mm_storeu_ps((float*)(tq+8*4),tw);_mm_storeu_ps((float*)(tq+12*4),tx);tq+=(16*
4);}}}static void hy(void*tn,void const*to,size_t tp){char*__restrict__ tq=(///
char*)to;char*__restrict__ tr=((char*)to)+tp;ptrdiff_t ts=(char*)tn-(char*)to;
if(ts>=16){char*__restrict__ tt=((char*)to)+(tp&~15);_Pragma("GCC unroll 1")///
_Pragma("GCC novector")do{__m128 tu;__asm__(""::"r"(tq));(tu)=_mm_loadu_ps((///
float const*)(tq));_mm_storeu_ps((float*)((tq+ts)),tu);tq+=16;}while(tq<tt);if(
tq==tr)return;}do{__asm__(""::"r"(tq));*(int*)(tq+ts)=*(int*)tq;tq+=4;}while(tq
<tr);}static float hz(float tn,float to,void*tp){float tq=to/2;float tr=0.5f+tq
;assert(to<=1);(void)sizeof(tp);if(tn<0.0f)tn=-tn;if(tn>=tr)return 0.0f;else{//
float ts=0.5f-tq;if(tn<=ts)return 1.0f;else return(tr-tn)/to;}}static float ib(
float tn,void*to){(void)sizeof(to);return 0.5f+tn/2.0f;}static float ic(float//
tn,float to,void*tp){(void)sizeof(to);(void)sizeof(tp);if(tn<0.0f)tn=-tn;if(tn
<=1.0f)return 1.0f-tn;else return 0.0f;}static float ie(float tn,float to,void*
tp){(void)sizeof(tn);(void)sizeof(to);(void)sizeof(tp);return 1.0f;}static/////
float ig(float tn,float to,void*tp){(void)sizeof(to);(void)sizeof(tp);if(tn<///
0.0f)tn=-tn;if(tn<1.0f)return(4.0f+tn*tn*(3.0f*tn-6.0f))/6.0f;else if(tn<2.0f)
return(8.0f+tn*(-12.0f+tn*(6.0f-tn)))/6.0f;return(0.0f);}static float ih(float
tn,float to,void*tp){(void)sizeof(to);(void)sizeof(tp);if(tn<0.0f)tn=-tn;if(tn<
1.0f)return 1.0f-tn*tn*(2.5f-1.5f*tn);else if(tn<2.0f)return 2.0f-tn*(4.0f+tn*(
0.5f*tn-2.5f));return(0.0f);}static float ii(float tn,float to,void*tp){(void)
sizeof(to);(void)sizeof(tp);if(tn<0.0f)tn=-tn;if(tn<1.0f)return(16.0f+tn*tn*(//
21.0f*tn-36.0f))/18.0f;else if(tn<2.0f)return(32.0f+tn*(-60.0f+tn*(36.0f-7.0f*
tn)))/18.0f;return(0.0f);}static float ij(float tn,void*to){(void)sizeof(tn);(
void)sizeof(to);return 0.5f;}static float ik(float tn,void*to){(void)sizeof(tn)
;(void)sizeof(to);return 1;}static float il(float tn,void*to){(void)sizeof(tn);
(void)sizeof(to);return 2;}static int im(fy*tn,float to,void*tp){assert(tn!=0);
if(to>=(1.0f-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20))))return
(int)hd(tn(1.0f/to,tp)*2.0f);else return(int)hd(tn(to,tp)*2.0f/to);}static int
io(gi*tn,int to,void*tp){float tq=tn->ad.ab;fy*tr=tn->ai;switch(to){case 1:////
return(int)hd(tr(1.0f/tq,tp)*2.0f);case 2:return(int)hd(tr(tq,tp)*2.0f/tq);case
0:return(int)hd(tr(tq,tp)*2.0f);default:assert((to>=0)&&(to<=2));return 0;}}///
static int ip(gi*tn,int to){if(to)return tn->ad.u;else return(tn->ad.l+tn->am*2
);}static int iq(int tn,int to){(void)sizeof(tn);(void)sizeof(to);return 0;}///
static int ir(int tn,int to){if(tn<0)return 0;if(tn>=to)return to-1;return tn;}
static int is(int tn,int to){if(tn<0){if(tn>-to)return-tn;else return to-1;}if(
tn>=to){int tp=to*2;if(tn>=tp)return 0;else return tp-tn-1;}return tn;}static//
int it(int tn,int to){if(tn>=0)return(tn%to);else{int tp=(-tn)%to;if(tp!=0)tp=
to-tp;return(tp);}}typedef int iu(int n,int max);static iu*iv[]={ir,is,it,iq,};
__inline__ static int iw(fs tn,int to,int tp){if(to>=0&&to<tp)return to;return
iv[tn](to,tp);}static void ix(gi*tn,gj*to){int tp,tq;int tr,ts;int tt=/////////
0x7fffffff,tu=-0x7fffffff;int tv=0x7fffffff,tw=-0x7fffffff;int tx=0x7fffffff,ty
=-0x7fffffff;fs tz=tn->aj;ge*ua=tn->l;int ub=tn->ad.u;int uc=tn->ad.l;int ud=tn
->am;assert(tn->ar);tq=ub;for(tp=0;tp<tq;tp++){assert(ua[tp].u>=ua[tp].l);if(ua
[tp].l<tt){tt=ua[tp].l;tq=tp+ud;if(tq>ub)tq=ub;}}tq=0;for(tp=ub-1;tp>=tq;tp--){
assert(ua[tp].u>=ua[tp].l);if(ua[tp].u>tu){tu=ua[tp].u;tq=tp-ud;if(tq<0)tq=0;}}
assert(to->l.l<=tt);assert(to->l.u>=tu);tr=0;if(tt<0){tr=-tt;tt=0;}ts=0;if(tu>=
uc){ts=tu-uc+1;tu=uc-1;}to->u[0]=tr;to->u[1]=ts;to->ab[0].l=tt;to->ab[0].u=tu;
to->ab[0].ab=tt;to->ab[1].l=0;to->ab[1].u=-1;to->ab[1].ab=0;if(tz==////////////
STBIR_EDGE_ZERO)return;for(tp=-tr;tp<0;tp++){int ue=iw(tz,tp,uc);if(ue<tv)tv=ue
;if(ue>tw)tw=ue;}for(tp=uc;tp<(uc+ts);tp++){int uf=iw(tz,tp,uc);if(uf<tx)tx=uf;
if(uf>ty)ty=uf;}if(tv!=0x7fffffff){if(((tv<=tt)&&((tw+16)>=tt))||((tt<=tv)&&((
tu+16)>=tw))){to->ab[0].l=tt=gq(tt,tv);to->ab[0].u=tu=gr(tu,tw);to->ab[0].ab=tt
;tr=0;}}if(tx!=0x7fffffff){if(((tx<=tt)&&((ty+16)>=tt))||((tt<=tx)&&((tu+16)>=
ty))){to->ab[0].l=tt=gq(tt,tx);to->ab[0].u=tu=gr(tu,ty);to->ab[0].ab=tt;ts=0;}}
assert(to->l.l<=tt);assert(to->l.u>=tu);if((tr)&&(tv!=0x7fffffff)){gg*ug=to->ab
+1;assert(ts==0);if(tv<to->ab[0].l){to->ab[1].ab=to->ab[0].l;to->ab[1].l=to->ab
[0].l;to->ab[1].u=to->ab[0].u;--ug;}ug->ab=tv;ug->l=-tr;ug->u=(tw-tv)-tr;to->u[
0]=0;}else if((ts)&&(tx!=0x7fffffff)){gg*uh=to->ab+1;if(tx<to->ab[0].l){to->ab[
1].ab=to->ab[0].l;to->ab[1].l=to->ab[0].l;to->ab[1].u=to->ab[0].u;--uh;}uh->ab=
tx;uh->l=to->ab[1].u+1;uh->u=to->ab[1].u+1+(ty-tx);to->u[1]=0;}if((to->ab[1].u>
to->ab[1].l)&&(to->ab[0].l>to->ab[1].l)){gg ui=to->ab[0];to->ab[0]=to->ab[1];to
->ab[1]=ui;}}static void iy(int*tn,int*to,float tp,float tq,float tr,float ts,
int tt,fs tu){int tv,tw;float tx=tp-tq;float ty=tp+tq;float tz=(tx+ts)*tr;float
ua=(ty+ts)*tr;tv=(int)(hb(tz+0.5f));tw=(int)(hb(ua-0.5f));if(tw<tv)tw=tv;if(tu
==STBIR_EDGE_WRAP){if(tv<-tt)tv=-tt;if(tw>=(tt*2))tw=(tt*2)-1;}*tn=tv;*to=tw;}
static void iz(float tn,fx*to,gh*tp,int tq,ge*tr,float*ts,int tt,fs tu,void*tv)
{int tw,tx;float ty=tp->ac;float tz=tp->ad;int ua=tp->l;int ub=tp->ag;int uc=((
tp->ae)&&(ub<tq));tx=tq;if(uc)tx=ub;for(tw=0;tw<tx;tw++){int ud;int ue;float uf
=(float)tw+0.5f;float ug=(uf+tz)*ty;int uh,ui;iy(&uh,&ui,uf,tn,ty,tz,ua,tu);if(
(ui-uh+1)>tt)ui=uh+tt-1;ue=-1;for(ud=0;ud<=ui-uh;ud++){float uj=(float)(ud+uh)+
0.5f;float uk=to(ug-uj,ty,tv);if(((uk<((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)
/(1<<20)/(1<<20)))&&(uk>-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<
20))))){if(ud==0){assert((ui-uh)!=0);++uh;ud--;continue;}uk=0;}else ue=ud;ts[ud
]=uk;}ui=ue+uh;tr->l=uh;tr->u=ui;assert(tr->u>=tr->l);++tr;ts+=tt;}}static void
ja(ge*tn,float*to,int tp,float tq,int tr){if(tn->u<tn->l){tn->l=tn->u=tp;to[0]=
tq;}else if(tp<=tn->u){if(tp<tn->l){if((tn->u-tp+1)<=tr){int ts,tt=tn->l-tp;for
(ts=tn->u-tn->l;ts>=0;ts--)to[ts+tt]=to[ts];for(ts=1;ts<tt;ts++)to[ts]=0;to[0]=
tq;tn->l=tp;}}else{to[tp-tn->l]+=tq;}}else{if((tp-tn->l+1)<=tr){int tu,tv=tp-tn
->l;for(tu=(tn->u-tn->l)+1;tu<tv;tu++)to[tu]=0;to[tv]=tq;tn->u=tp;}}}static////
void jb(int*tn,int*to,float tp,float tq,float tr,float ts,int tt){float tu=tp-
tq;float tv=tp+tq;float tw=tu*tr-ts;float tx=tv*tr-ts;int ty=(int)(hb(tw+0.5f))
;int tz=(int)(hb(tx-0.5f));if(ty<0)ty=0;if(tz>=tt)tz=tt-1;*tn=ty;*to=tz;}static
void jc(int tn,int to,float tp,fx*tq,gh*tr,int ts,int tt,ge*tu,float*tv,void*tw
){int tx;int ty;int tz=-1;float ua=tr->ab;float ub=tr->ad;int uc=tr->u;int ud=
tr->ag;int ue=((tr->ae)&&(ud<uc));(void)sizeof(tt);for(tx=tn;tx<to;tx++){float
uf=(float)tx+0.5f;float ug=uf*ua-ub;int uh,ui;jb(&uh,&ui,uf,tp,ua,ub,uc);if(uh>
ui)continue;if(ue){if(uh==ud)break;if(ui>=ud)ui=ud-1;}for(ty=0;ty<=ui-uh;ty++){
float uj=(float)(ty+uh)+0.5f;float uk=uj-ug;float ul=tq(uk,ua,tw)*ua;if(((ul<((
float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))&&(ul>-((float)1/(1<<
20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))))ul=0.0f;{int um=ty+uh;float*un=
tv+um*ts;ge*uo=tu+um;if(um>tz){assert(um==(tz+1));tz=um;uo->l=tx;uo->u=tx;un[0]
=ul;}else{if(un[0]==0.0f){assert((tx-uo->l)==1);uo->l=tx;}uo->u=tx;assert((tx-
uo->l)<ts);un[tx-uo->l]=ul;}}}}}static void jd(fs tn,gf*to,gh*tp,int tq,ge*tr,
float*ts,int tt){int tu=tp->l;int tv=tu-1;int tw,tx;int ty=0x7fffffff;int tz=-
0x7fffffff;int ua=-1;int ub=tp->ag;int uc=tp->ah;int ud=((tp->ae)&&(ub<tq));///
float*ue;ge*uf;ue=ts;uf=tr;tx=tq;if(ud)tx=ub;for(tw=0;tw<tx;tw++){int ug;double
uh,ui=0;int uj;uj=uf->u-uf->l;for(ug=0;ug<=uj;ug++){ui+=(double)ue[ug];assert((
ue[ug]>=-2.0f)&&(ue[ug]<=2.0f));}if((ui<((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<
20)/(1<<20)/(1<<20)))&&(ui>-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(
1<<20)))){uf->u=uf->l;ue[0]=0.0f;}else{if((ui<(1.0f-((float)1/(1<<20)/(1<<20)/(
1<<20)/(1<<20)/(1<<20)/(1<<20))))||(ui>(1.0f+((float)1/(1<<20)/(1<<20)/(1<<20)/
(1<<20)/(1<<20)/(1<<20))))){uh=((double)1.0)/ui;for(ug=0;ug<=uj;ug++)ue[ug]=(//
float)(ue[ug]*uh);}}++uf;ue+=tt;}if(ud){ge*uk=tr;ge*ul=tr+ub;for(tw=ub;tw<tq;tw
++){ul->l=uk->l+uc;ul->u=uk->u+uc;++ul;++uk;}hy(ts+ub*tt,ts,(tq-ub)*tt*sizeof(
ue[0]));}ue=ts;uf=tr;for(tw=0;tw<tq;tw++){int um;if(tn==STBIR_EDGE_ZERO){if(uf
->u>tv)uf->u=tv;if(uf->l<0){int un,uo,up=0;up=-uf->l;uf->l=0;uo=uf->u-uf->l+1;
if(uo>0){for(un=0;un<uo;un++)ue[un]=ue[un+up];}}}else if((tn==STBIR_EDGE_CLAMP)
||(tn==STBIR_EDGE_REFLECT)){if(uf->u>tv){int uq=uf->l;int ur=uf->u;uf->u=tv;for
(um=tu;um<=ur;um++)ja(uf,ue,iv[tn](um,tu),ue[um-uq],tt);}if(uf->l<0){int us;///
float ut;float*uu=ue-(uf->l+1);for(um=-1;um>uf->l;um--)ja(uf,ue,iv[tn](um,tu),*
uu--,tt);us=uf->l;ut=uu[0];uf->l=0;for(um=0;um<=uf->u;um++)ue[um]=ue[um-us];ja(
uf,ue,iv[tn](us,tu),ut,tt);}}if(uf->l<=uf->u){int uv=uf->u-uf->l+1;while(uv&&(
ue[uv-1]==0.0f))--uv;uf->u=uf->l+uv-1;if(uf->l<=uf->u){if(uf->l<ty)ty=uf->l;if(
uf->u>tz)tz=uf->u;if(uv>ua)ua=uv;}for(um=uv;um<tt;um++)ue[um]=0.0f;}++uf;ue+=tt
;}to->l=ty;to->u=tz;to->ab=ua;}static int je(int tn,ge*to,float*tp,int tq,int//
tr,int ts,int tt){int tu=tt+1;(void)sizeof(ts);if(tq!=tr){float*tv=tp;float*tw=
tp;float*tx=tp+tn*tr;switch(tr){case 1:_Pragma("GCC unroll 1")_Pragma(/////////
"GCC novector")do{{__asm__(""::"r"(tv));((fp*)(tv))[0]=((fp*)(tw))[0];};++tv;tw
+=tq;}while(tv<tx);break;case 2:_Pragma("GCC unroll 1")_Pragma("GCC novector")
do{{__asm__(""::"r"(tv));((fq*)(tv))[0]=((fq*)(tw))[0];};tv+=2;tw+=tq;}while(tv
<tx);break;case 3:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{__asm__(""
::"r"(tv));((fq*)(tv))[0]=((fq*)(tw))[0];};{__asm__(""::"r"(tv+2));((fp*)(tv+2)
)[0]=((fp*)(tw+2))[0];};tv+=3;tw+=tq;}while(tv<tx);break;case 4:_Pragma(///////
"GCC unroll 1")_Pragma("GCC novector")do{{__m128 ty;__asm__(""::"r"(tv));(ty)=
_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),ty);};tv+=4;tw+=tq;
}while(tv<tx);break;case 5:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{//
__m128 tz;__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float const*)(tw));//////////
_mm_storeu_ps((float*)(tv),tz);};{__asm__(""::"r"(tv+4));((fp*)(tv+4))[0]=((fp*
)(tw+4))[0];};tv+=5;tw+=tq;}while(tv<tx);break;case 6:_Pragma("GCC unroll 1")//
_Pragma("GCC novector")do{{__m128 ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));_mm_storeu_ps((float*)(tv),ua);};{__asm__(""::"r"(tv+4));((
fq*)(tv+4))[0]=((fq*)(tw+4))[0];};tv+=6;tw+=tq;}while(tv<tx);break;case 7://///
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{__m128 ub;__asm__(""::"r"(tv)
);(ub)=_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),ub);};{/////
__asm__(""::"r"(tv+4));((fq*)(tv+4))[0]=((fq*)(tw+4))[0];};{__asm__(""::"r"(tv+
6));((fp*)(tv+6))[0]=((fp*)(tw+6))[0];};tv+=7;tw+=tq;}while(tv<tx);break;case 8
:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{__m128 uc;__asm__(""::"r"(tv
));(uc)=_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),uc);};{////
__m128 ud;__asm__(""::"r"(tv+4));(ud)=_mm_loadu_ps((float const*)(tw+4));//////
_mm_storeu_ps((float*)(tv+4),ud);};tv+=8;tw+=tq;}while(tv<tx);break;case 9:////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{__m128 ue;__asm__(""::"r"(tv)
);(ue)=_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),ue);};{/////
__m128 uf;__asm__(""::"r"(tv+4));(uf)=_mm_loadu_ps((float const*)(tw+4));//////
_mm_storeu_ps((float*)(tv+4),uf);};{__asm__(""::"r"(tv+8));((fp*)(tv+8))[0]=((
fp*)(tw+8))[0];};tv+=9;tw+=tq;}while(tv<tx);break;case 10:_Pragma(/////////////
"GCC unroll 1")_Pragma("GCC novector")do{{__m128 ug;__asm__(""::"r"(tv));(ug)=
_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),ug);};{__m128 uh;//
__asm__(""::"r"(tv+4));(uh)=_mm_loadu_ps((float const*)(tw+4));_mm_storeu_ps((
float*)(tv+4),uh);};{__asm__(""::"r"(tv+8));((fq*)(tv+8))[0]=((fq*)(tw+8))[0];}
;tv+=10;tw+=tq;}while(tv<tx);break;case 11:_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")do{{__m128 ui;__asm__(""::"r"(tv));(ui)=_mm_loadu_ps((float/////
const*)(tw));_mm_storeu_ps((float*)(tv),ui);};{__m128 uj;__asm__(""::"r"(tv+4))
;(uj)=_mm_loadu_ps((float const*)(tw+4));_mm_storeu_ps((float*)(tv+4),uj);};{//
__asm__(""::"r"(tv+8));((fq*)(tv+8))[0]=((fq*)(tw+8))[0];};{__asm__(""::"r"(tv+
10));((fp*)(tv+10))[0]=((fp*)(tw+10))[0];};tv+=11;tw+=tq;}while(tv<tx);break;//
case 12:_Pragma("GCC unroll 1")_Pragma("GCC novector")do{{__m128 uk;__asm__(""
::"r"(tv));(uk)=_mm_loadu_ps((float const*)(tw));_mm_storeu_ps((float*)(tv),uk)
;};{__m128 ul;__asm__(""::"r"(tv+4));(ul)=_mm_loadu_ps((float const*)(tw+4));//
_mm_storeu_ps((float*)(tv+4),ul);};{__m128 um;__asm__(""::"r"(tv+8));(um)=/////
_mm_loadu_ps((float const*)(tw+8));_mm_storeu_ps((float*)(tv+8),um);};tv+=12;tw
+=tq;}while(tv<tx);break;default:_Pragma("GCC unroll 1")_Pragma("GCC novector")
do{float*un=tv+tr-4;float*uo=tw;do{__asm__(""::"r"(tv));{__m128 up;__asm__(""::
"r"(tv));(up)=_mm_loadu_ps((float const*)(uo));_mm_storeu_ps((float*)(tv),up);}
;tv+=4;uo+=4;}while(tv<=un);un+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector"
)while(tv<un){{__asm__(""::"r"(tv));((fp*)(tv))[0]=((fp*)(uo))[0];};++tv;++uo;}
tw+=tq;}while(tv<tx);break;}}tp[tr*tn]=8888.0f;{ge*uq=to+tn-1;float*ur=tp+tr*(
tn-1);while((uq>=to)&&((uq->l+tr*2)>=tu)){if((uq->l+tr)>tu){int us=tr;if(tr>12)
{int ut;ut=tr&3;us=(((uq->u-uq->l+1)-ut+3)&~3)+ut;if(us<(8+ut))us=8+ut;}if((uq
->l+us)>tu){int uu=tu-us;int uv=uq->u-uq->l+1;int uw=uq->l-uu;float*ux=ur+uv-1;
float*uy=ux+uw;assert((uu>=ts)&&(uu<uq->l));while(uv){*uy-- =*ux--;--uv;}while(
uy>=ur)*uy-- =0;uq->l=uu;if(tr>12){int uz;uz=tr&3;us=(((uq->u-uq->l+1)-uz+3)&~3
)+uz;if(us<(8+uz))us=8+uz;}}}--uq;ur-=tr;}}return tr;}static void jf(gi*tn,gi*
to,void*tp){int tq;float tr=tn->ad.ab;fx*ts=tn->ah;fy*tt=tn->ai;float tu=tn->ad
.ac;int tv=tn->ad.l;int tw=tn->an;ge*tx=tn->l;float*ty=tn->u;int tz=tn->ak;////
switch(tn->ar){case 1:{float ua=tt(tu,tp)*tr;iz(ua,ts,&tn->ad,tw,tx,ty,tz,tn->
aj,tp);;jd(tn->aj,&tn->aq,&tn->ad,tw,tx,ty,tz);;}break;case 0:case 2:{float ub=
tt(tr,tp)*tu;int uc=tn->am;int ud=tv+uc;if(!tn->ar){if(to){tx=to->l;ty=to->u;tz
=to->ak;tw=to->an;tn->aq.l=to->aq.l;tn->aq.u=to->aq.u;tn->aq.ab=to->aq.ab;goto
jump_right_to_pivot;}tx=tn->ab;ty=tn->ac;tz=tn->at;tw=tn->as;}jc(-uc,ud,ub,ts,&
tn->ad,tz,tw,tx,ty,tp);;jd(tn->aj,&tn->aq,&tn->ad,tw,tx,ty,tz);;if(!tn->ar){ge*
ue;int uf;jump_right_to_pivot:;uf=(-uc)-1;for(tq=0;tq<tw;tq++){int ug;int uh=tx
->l,ui=tx->u;int uj=tn->ak;float*uk=tn->u+(uh+uc)*uj;float*ul=ty;ue=tn->l+(uh+
uc);for(ug=uh;ug<=ui;ug++){float um=*ul++;if(((um>=((float)1/(1<<20)/(1<<20)/(1
<<20)/(1<<20)/(1<<20)/(1<<20)))||(um<=-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20
)/(1<<20)/(1<<20))))){if((ug>uf)||(ue->l>ue->u)){{ge*un=tn->l+(uf+uc+1);while(
un<ue){un->l=0;un->u=-1;++un;}}ue->l=tq;ue->u=tq;uk[0]=um;uf=ug;}else{ja(ue,uk,
tq,um,uj);}assert((ue->u-ue->l+1)<=uj);}++ue;uk+=uj;}++tx;ty+=tz;}{ge*uo=tn->l+
(uf+uc+1);ge*up=tn->l+tn->an;while(uo<up){uo->l=0;uo->u=-1;++uo;}};}}break;}}//
static float*jg(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*
tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned char
const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128 tz,ua
,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i
ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8(tu,ud);
tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=_mm_unpacklo_epi16
(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(tv);(ua)=//////////
_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);(tz)=////
_mm_mul_ps(tz,(ho));(ua)=_mm_mul_ps(ua,(ho));(ub)=_mm_mul_ps(ub,(ho));(uc)=////
_mm_mul_ps(uc,(ho));;;;;_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)
(tq+4),ua);_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);
tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16
;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""
::"r"(tq));tq[0-4]=((float)(ts[0]))*3.9215689e-03f;tq[1-4]=((float)(ts[1]))*///
3.9215689e-03f;tq[2-4]=((float)(ts[2]))*3.9215689e-03f;tq[3-4]=((float)(ts[3]))
*3.9215689e-03f;tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=((float)(ts[0]))*///////
3.9215689e-03f;tq+=1;ts+=1;}return tr;}static void jh(void*tn,int to,float/////
const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((
unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){////
__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((
hm),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hm),///
_mm_loadu_ps((float const*)(tp+4))));;;{__m128 tw,tx;__m128i ty,tz;tw=/////////
_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,_mm_setzero_ps());tx=//
_mm_max_ps(tx,_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx)
;ty=_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};_mm_storel_epi64((/////
__m128i*)(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq
=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq
<=tr){__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*
)(tp));(ua)=_mm_add_ps((hq),_mm_mul_ps((hm),ua));;{__m128 uc,ud;__m128i ue,uf;
uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,_mm_setzero_ps());ud
=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud
);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=////////
_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(///////
"GCC novector")while(tq<tr){__m128 ug;__asm__(""::"r"(tp));(ug)=_mm_add_ss((hq)
,_mm_mul_ss((hm),_mm_load_ss((float const*)(tp+0))));tq[0]=((unsigned char)////
_mm_cvtsi128_si32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ug,(hm)),/////////////
_mm_setzero_ps()))));tq+=1;tp+=1;}}static float*ji(float*tn,int to,void const*
tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*ts=(///
unsigned char const*)tp;unsigned char const*tt=ts+to-16;if(to>=16){tr-=16;for(;
;){__m128i tu,tv,tw,tx,ty;__m128 tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=////////
_mm_loadu_si128((__m128i const*)(ts));{__m128i ud=_mm_setzero_si128();tx=//////
_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8(tu,ud);tv=_mm_unpacklo_epi16(tx,
ud);tw=_mm_unpackhi_epi16(tx,ud);tx=_mm_unpacklo_epi16(ty,ud);ty=//////////////
_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(tv);(ua)=_mm_cvtepi32_ps(tw);(
ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);;;;;_mm_storeu_ps((float*)(tq+
0),tz);_mm_storeu_ps((float*)(tq+4),ua);_mm_storeu_ps((float*)(tq+8),ub);//////
_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=((float)(ts[0]));tq[1
-4]=((float)(ts[1]));tq[2-4]=((float)(ts[2]));tq[3-4]=((float)(ts[3]));tq+=4;ts
+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(
""::"r"(tq));tq[0]=((float)(ts[0]));tq+=1;ts+=1;}return tr;}static void jj(void
*tn,int to,float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;///
unsigned char*tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;
tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((
hq),_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float
const*)(tp+4)));;;{__m128 tw,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=/////////
_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,_mm_setzero_ps());tx=_mm_max_ps(tx,/////////
_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx);ty=//////////
_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};_mm_storel_epi64((__m128i*)
(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts
;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){///
__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*)(tp))
;(ua)=_mm_add_ps((hq),ua);;{__m128 uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=
_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,_mm_setzero_ps());ud=_mm_max_ps(ud,/////////
_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud);ue=//////////
_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=/////////////
_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(///////
"GCC novector")while(tq<tr){float ug;__asm__(""::"r"(tp));ug=tp[0]+0.5f;for(;;)
{if((ug)<(0))(ug)=(0);if((ug)>(255))(ug)=(255);break;};tq[0]=(unsigned char)ug;
tq+=1;tp+=1;}}static float*jk(float*tn,int to,void const*tp){float*__restrict__
tq=tn;float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;tq
+=4;while(tq<=tr){tq[0-4]=gs[ts[0]];tq[1-4]=gs[ts[1]];tq[2-4]=gs[ts[2]];tq[3-4]
=gs[ts[3]];tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")///
while(tq<tr){__asm__(""::"r"(tq));tq[0]=gs[ts[0]];tq+=1;ts+=1;}return tr;}/////
static void jl(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(//
unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=16){float////
const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__
(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=_mm_loadu_ps((float
const*)((tp)+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((
float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=//////
_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=
_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=///////
_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=////////
_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=_mm_srli_epi32(_mm_castps_si128(tt),20
);;(tu)=_mm_max_ps(tu,_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,//////////////
_mm_castsi128_ps((hu)));ty=_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=//////
_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu))
);tz=_mm_srli_epi32(_mm_castps_si128(tv),20);;(tw)=_mm_max_ps(tw,//////////////
_mm_castsi128_ps((ht)));(tw)=_mm_min_ps(tw,_mm_castsi128_ps((hu)));ua=/////////
_mm_srli_epi32(_mm_castps_si128(tw),20);;{hk uf,ug,uh,ui;uf.ab=tx;ug.ab=ty;uh.
ab=tz;ui.ab=ua;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]
];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu
-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[
ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh.l
[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(127-
13)*8)[uh.u[3]];ui.l[0]=(gu-(127-13)*8)[ui.u[0]];ui.l[1]=(gu-(127-13)*8)[ui.u[1
]];ui.l[2]=(gu-(127-13)*8)[ui.u[2]];ui.l[3]=(gu-(127-13)*8)[ui.u[3]];tx=uf.ab;
ty=ug.ab;tz=uh.ab;ua=ui.ab;};{__m128i uj;uj=_mm_srli_epi32(_mm_castps_si128(tt)
,12);(uj)=_mm_and_si128(uj,(hv));(uj)=_mm_or_si128(uj,(hw));(tx)=_mm_madd_epi16
(tx,uj);tx=_mm_srli_epi32(tx,16);};{__m128i uk;uk=_mm_srli_epi32(//////////////
_mm_castps_si128(tu),12);(uk)=_mm_and_si128(uk,(hv));(uk)=_mm_or_si128(uk,(hw))
;(ty)=_mm_madd_epi16(ty,uk);ty=_mm_srli_epi32(ty,16);};{__m128i ul;ul=/////////
_mm_srli_epi32(_mm_castps_si128(tv),12);(ul)=_mm_and_si128(ul,(hv));(ul)=//////
_mm_or_si128(ul,(hw));(tz)=_mm_madd_epi16(tz,ul);tz=_mm_srli_epi32(tz,16);};{//
__m128i um;um=_mm_srli_epi32(_mm_castps_si128(tw),12);(um)=_mm_and_si128(um,(hv
));(um)=_mm_or_si128(um,(hw));(ua)=_mm_madd_epi16(ua,um);ua=_mm_srli_epi32(ua,
16);};tx=_mm_packs_epi32(tx,ty);tz=_mm_packs_epi32(tz,ua);ty=_mm_unpacklo_epi16
(tx,tz);ua=_mm_unpackhi_epi16(tx,tz);tx=_mm_unpacklo_epi16(ty,ua);tz=//////////
_mm_unpackhi_epi16(ty,ua);tx=_mm_packus_epi16(tx,tz);_mm_storeu_si128((__m128i*
)(tq),tx);;tp+=16;tq+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){/////
__asm__(""::"r"(tp));tq[0-4]=gv(tp[0]);tq[1-4]=gv(tp[1]);tq[2-4]=gv(tp[2]);tq[3
-4]=gv(tp[3]);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")
while(tq<tr){__asm__(""::"r"(tp));tq[0]=gv(tp[0]);tq+=1;tp+=1;}}static float*jm
(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to
;unsigned char const*ts=(unsigned char const*)tp;do{tq[0]=gs[ts[0]];tq[1]=gs[ts
[1]];tq[2]=gs[ts[2]];tq[3]=((float)ts[3])*3.9215689e-03f;ts+=4;tq+=4;}while(tq<
tr);return tr;}static void jo(void*tn,int to,float const*tp){unsigned char*////
__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(
to>=16){float const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty
,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=//////
_mm_loadu_ps((float const*)((tp)+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));
(tw)=_mm_loadu_ps((float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=/////////////
_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=
_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=/////
_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,/////////////
_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=/////////
_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,_mm_castsi128_ps((
ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=_mm_srli_epi32(////////////
_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=/////
_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(_mm_castps_si128(tv),20
);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=_mm_max_ps(tw,_mm_setzero_ps(
));(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(tw);;{hk uf,ug,uh;uf.ab=tx;ug
.ab=ty;uh.ab=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1
]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(
gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8
)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh
.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(//
127-13)*8)[uh.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;};{__m128i ui;ui=_mm_srli_epi32(
_mm_castps_si128(tt),12);(ui)=_mm_and_si128(ui,(hv));(ui)=_mm_or_si128(ui,(hw))
;(tx)=_mm_madd_epi16(tx,ui);tx=_mm_srli_epi32(tx,16);};{__m128i uj;uj=/////////
_mm_srli_epi32(_mm_castps_si128(tu),12);(uj)=_mm_and_si128(uj,(hv));(uj)=//////
_mm_or_si128(uj,(hw));(ty)=_mm_madd_epi16(ty,uj);ty=_mm_srli_epi32(ty,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tv),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(tz)=_mm_madd_epi16(tz,uk);tz=_mm_srli_epi32(tz,
16);};tx=_mm_packs_epi32(tx,ty);tz=_mm_packs_epi32(tz,ua);ty=_mm_unpacklo_epi16
(tx,tz);ua=_mm_unpackhi_epi16(tx,tz);tx=_mm_unpacklo_epi16(ty,ua);tz=//////////
_mm_unpackhi_epi16(ty,ua);tx=_mm_packus_epi16(tx,tz);_mm_storeu_si128((__m128i*
)(tq),tx);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float ul;__asm__(""::
"r"(tp));tq[0]=gv(tp[0]);tq[1]=gv(tp[1]);tq[2]=gv(tp[2]);ul=tp[3]*255.0f+0.5f;
for(;;){if((ul)<(0))(ul)=(0);if((ul)>(255))(ul)=(255);break;};tq[3]=(unsigned//
char)ul;tq+=4;tp+=4;}while(tq<tr);}static float*jp(float*tn,int to,void const*
tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*ts=(///
unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[0]];tq[1-4]=((float)
ts[1])*3.9215689e-03f;tq[2-4]=gs[ts[0+2]];tq[3-4]=((float)ts[1+2])*////////////
3.9215689e-03f;ts+=4;tq+=4;}tq-=4;if(tq<tr){tq[0]=gs[ts[0]];tq[1]=((float)ts[1]
)*3.9215689e-03f;}return tr;}static void jq(void*tn,int to,float const*tp){////
unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((unsigned///
char*)tq)+to;if(to>=16){float const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,
tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((
tp)));(tu)=_mm_loadu_ps((float const*)((tp)+4));(tv)=_mm_loadu_ps((float const*
)((tp)+8));(tw)=_mm_loadu_ps((float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=//
_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=
_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=/////
_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,/////////////
_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=/////////
_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_add_ps((hq),_mm_mul_ps((hm),
tu));(tu)=_mm_max_ps(tu,_mm_setzero_ps());(tu)=_mm_min_ps(tu,(hm));(ty)=///////
_mm_cvttps_epi32(tu);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=/////////
_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(_mm_castps_si128(tv),20
);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=_mm_max_ps(tw,_mm_setzero_ps(
));(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(tw);;{hk uf,ug;uf.ab=tx;ug.ab
=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(
gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8
)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug
.l[3]=(gu-(127-13)*8)[ug.u[3]];tx=uf.ab;tz=ug.ab;};{__m128i uh;uh=/////////////
_mm_srli_epi32(_mm_castps_si128(tt),12);(uh)=_mm_and_si128(uh,(hv));(uh)=//////
_mm_or_si128(uh,(hw));(tx)=_mm_madd_epi16(tx,uh);tx=_mm_srli_epi32(tx,16);};{//
__m128i ui;ui=_mm_srli_epi32(_mm_castps_si128(tv),12);(ui)=_mm_and_si128(ui,(hv
));(ui)=_mm_or_si128(ui,(hw));(tz)=_mm_madd_epi16(tz,ui);tz=_mm_srli_epi32(tz,
16);};tx=_mm_packs_epi32(tx,ty);tz=_mm_packs_epi32(tz,ua);ty=_mm_unpacklo_epi16
(tx,tz);ua=_mm_unpackhi_epi16(tx,tz);tx=_mm_unpacklo_epi16(ty,ua);tz=//////////
_mm_unpackhi_epi16(ty,ua);tx=_mm_packus_epi16(tx,tz);_mm_storeu_si128((__m128i*
)(tq),tx);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float uj;__asm__(""::
"r"(tp));tq[0]=gv(tp[0]);uj=tp[1]*255.0f+0.5f;for(;;){if((uj)<(0))(uj)=(0);if((
uj)>(255))(uj)=(255);break;};tq[1]=(unsigned char)uj;tq+=2;tp+=2;}while(tq<tr);
}static float*jr(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*
tr=(float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;unsigned////
short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128 tx,ty;//
__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i tz=///
_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz);}
;(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_mul_ps(tx,(hp));(ty
)=_mm_mul_ps(ty,(hp));;;_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)
(tq+4),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}////
return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){
__asm__(""::"r"(tq));tq[0-4]=((float)(ts[0]))*1.5259022e-05f;tq[1-4]=((float)(
ts[1]))*1.5259022e-05f;tq[2-4]=((float)(ts[2]))*1.5259022e-05f;tq[3-4]=((float)
(ts[3]))*1.5259022e-05f;tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=((float)(ts[0]))*///////
1.5259022e-05f;tq+=1;ts+=1;}return tr;}static void js(void*tn,int to,float/////
const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned short*tr=
((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){
__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((
hn),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hn),///
_mm_loadu_ps((float const*)(tp+4))));;;{__m128i tw,tx;tw=_mm_cvttps_epi32(/////
_mm_max_ps(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=_mm_cvttps_epi32(/////////
_mm_max_ps(_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=//
_mm_sub_epi32(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};//////
_mm_storeu_si128((__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(
tr+4*2))break;tq=tr;tp=ts;}return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(tq<=tr){__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=///
_mm_loadu_ps((float const*)(tp));(ty)=_mm_add_ps((hq),_mm_mul_ps((hn),ty));;{//
__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps
()));ub=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=
_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=/////
_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq
-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__m128 uc;//////
__asm__(""::"r"(tp));(uc)=_mm_add_ss((hq),_mm_mul_ss((hn),_mm_load_ss((float///
const*)(tp+0))));tq[0]=((unsigned short)_mm_cvtsi128_si32(_mm_cvttps_epi32(////
_mm_max_ps(_mm_min_ps(uc,(hn)),_mm_setzero_ps()))));tq+=1;tp+=1;}}static float*
jt(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+
to;unsigned short const*ts=(unsigned short const*)tp;unsigned short const*tt=ts
+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128 tx,ty;__asm__(""::"r"(tq)
);(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i tz=_mm_setzero_si128();tv
=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz);};(tx)=_mm_cvtepi32_ps(
tv);(ty)=_mm_cvtepi32_ps(tw);;;_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((
float*)(tq+4),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=
tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=
tr){__asm__(""::"r"(tq));tq[0-4]=((float)(ts[0]));tq[1-4]=((float)(ts[1]));tq[2
-4]=((float)(ts[2]));tq[3-4]=((float)(ts[3]));tq+=4;ts+=4;}tq-=4;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=(
(float)(ts[0]));tq+=1;ts+=1;}return tr;}static void ju(void*tn,int to,float////
const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned short*tr=
((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){
__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_loadu_ps(
(float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp+4)));;
;{__m128i tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),////////////
_mm_setzero_ps()));tx=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),/////////
_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=/////////
_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq
),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}///
return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){////
__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp))
;(ty)=_mm_add_ps((hq),ty);;{__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(//////
_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=_mm_cvttps_epi32(_mm_max_ps(/////////
_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32
(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=_mm_sub_epi16(tz,gz);};_mm_storel_epi64((
__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(///////
"GCC novector")while(tq<tr){float uc;__asm__(""::"r"(tp));uc=tp[0]+0.5f;for(;;)
{if((uc)<(0))(uc)=(0);if((uc)>(65535))(uc)=(65535);break;};tq[0]=(unsigned/////
short)uc;tq+=1;tp+=1;}}static float*jv(float*tn,int to,void const*tp){float*///
__restrict__ tq=tn;float*tr=(float*)tq+to;he const*ts=(he const*)tp;if(to>=8){
he const*tt=ts+to-8;tr-=8;for(;;){__asm__(""::"r"(tq));hh(tq,ts);tq+=8;ts+=8;if
(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;_Pragma(//
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4
]=hf(ts[0]);tq[1-4]=hf(ts[1]);tq[2-4]=hf(ts[2]);tq[3-4]=hf(ts[3]);tq+=4;ts+=4;}
tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__("":://
"r"(tq));tq[0]=hf(ts[0]);tq+=1;ts+=1;}return tr;}static void jw(void*tn,int to,
float const*tp){he*__restrict__ tq=(he*)tn;he*tr=((he*)tq)+to;if(to>=8){float//
const*ts=tp+to-8;tr-=8;for(;;){__asm__(""::"r"(tp));hj(tq,tp);tp+=8;tq+=8;if(tq
<=tr)continue;if(tq==(tr+8))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4
]=hg(tp[0]);tq[1-4]=hg(tp[1]);tq[2-4]=hg(tp[2]);tq[3-4]=hg(tp[3]);tq+=4;tp+=4;}
tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__("":://
"r"(tq));tq[0]=hg(tp[0]);tq+=1;tp+=1;}}static float*jx(float*tn,int to,void////
const*tp){if((void*)tn!=tp)hx(tn,tp,to*sizeof(float));return tn+to;}static void
jy(void*tn,int to,float const*tp){if((void*)tn!=(void*)tp)hx(tn,tp,to*sizeof(//
float));}static float*jz(float*tn,int to,void const*tp){float*__restrict__ tq=
tn;float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;/////
unsigned char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,
ty;__m128 tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*
)(ts));{__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=/////////
_mm_unpackhi_epi8(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,
ud);tx=_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=//////////
_mm_cvtepi32_ps(tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=////
_mm_cvtepi32_ps(ty);(tz)=_mm_mul_ps(tz,(ho));(ua)=_mm_mul_ps(ua,(ho));(ub)=////
_mm_mul_ps(ub,(ho));(uc)=_mm_mul_ps(uc,(ho));(tz)=_mm_castsi128_ps(////////////
_mm_shuffle_epi32(_mm_castps_si128(tz),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(1<<
2)+(0<<4)+(3<<6)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc)
,(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((
float*)(tq+4),ua);_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12
),uc);tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return
tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){//////
__asm__(""::"r"(tq));tq[0-4]=((float)(ts[2]))*3.9215689e-03f;tq[1-4]=((float)(
ts[1]))*3.9215689e-03f;tq[2-4]=((float)(ts[0]))*3.9215689e-03f;tq[3-4]=((float)
(ts[3]))*3.9215689e-03f;tq+=4;ts+=4;}tq-=4;return tr;}static void ka(void*tn,//
int to,float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;///////
unsigned char*tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;
tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((
hq),_mm_mul_ps((hm),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),///
_mm_mul_ps((hm),_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(////
_mm_shuffle_epi32(_mm_castps_si128(tt),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(tu)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));{__m128 tw,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw
=_mm_max_ps(tw,_mm_setzero_ps());tx=_mm_max_ps(tx,_mm_setzero_ps());ty=////////
_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx);ty=_mm_packs_epi32(ty,tz);tv=/////
_mm_packus_epi16(ty,ty);};_mm_storel_epi64((__m128i*)(tq),(tv));tp+=4*2;tq+=4*2
;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma(//
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__m128 ua;__m128i ub;//////
__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*)(tp));(ua)=_mm_add_ps((hq)
,_mm_mul_ps((hm),ua));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
ua),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128 uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,
hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,_mm_setzero_ps());ud=_mm_max_ps(ud,//
_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud);ue=//////////
_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=/////////////
_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;}static float*kb(float*tn,int to,void
const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*
ts=(unsigned char const*)tp;unsigned char const*tt=ts+to-16;if(to>=16){tr-=16;
for(;;){__m128i tu,tv,tw,tx,ty;__m128 tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=///
_mm_loadu_si128((__m128i const*)(ts));{__m128i ud=_mm_setzero_si128();tx=//////
_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8(tu,ud);tv=_mm_unpacklo_epi16(tx,
ud);tw=_mm_unpackhi_epi16(tx,ud);tx=_mm_unpacklo_epi16(ty,ud);ty=//////////////
_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(tv);(ua)=_mm_cvtepi32_ps(tw);(
ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);(tz)=_mm_castsi128_ps(////////
_mm_shuffle_epi32(_mm_castps_si128(tz),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(1<<
2)+(0<<4)+(3<<6)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc)
,(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((
float*)(tq+4),ua);_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12
),uc);tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return
tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){//////
__asm__(""::"r"(tq));tq[0-4]=((float)(ts[2]));tq[1-4]=((float)(ts[1]));tq[2-4]=
((float)(ts[0]));tq[3-4]=((float)(ts[3]));tq+=4;ts+=4;}tq-=4;return tr;}static
void kc(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(unsigned
char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp
+to-4*2;tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=/////
_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),////////
_mm_loadu_ps((float const*)(tp+4)));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(tt),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(tu)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128 tw
,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,//
_mm_setzero_ps());tx=_mm_max_ps(tx,_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz
=_mm_cvttps_epi32(tx);ty=_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};//
_mm_storel_epi64((__m128i*)(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq
==(tr+4*2))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=///
_mm_loadu_ps((float const*)(tp));(ua)=_mm_add_ps((hq),ua);(ua)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128//
uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,
_mm_setzero_ps());ud=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf
=_mm_cvttps_epi32(ud);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(
int*)(tq-4)=_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;}static float*kd(float*tn,
int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned
char const*ts=(unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[2]];tq
[1-4]=gs[ts[1]];tq[2-4]=gs[ts[0]];tq[3-4]=gs[ts[3]];tq+=4;ts+=4;}tq-=4;return//
tr;}static void ke(void*tn,int to,float const*tp){unsigned char*__restrict__ tq
=(unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=16){float//
const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__
(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=_mm_loadu_ps((float
const*)((tp)+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((
float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=//////
_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=
_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=///////
_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=////////
_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=_mm_srli_epi32(_mm_castps_si128(tt),20
);;(tu)=_mm_max_ps(tu,_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,//////////////
_mm_castsi128_ps((hu)));ty=_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=//////
_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu))
);tz=_mm_srli_epi32(_mm_castps_si128(tv),20);;(tw)=_mm_max_ps(tw,//////////////
_mm_castsi128_ps((ht)));(tw)=_mm_min_ps(tw,_mm_castsi128_ps((hu)));ua=/////////
_mm_srli_epi32(_mm_castps_si128(tw),20);;{hk uf,ug,uh,ui;uf.ab=tx;ug.ab=ty;uh.
ab=tz;ui.ab=ua;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]
];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu
-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[
ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh.l
[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(127-
13)*8)[uh.u[3]];ui.l[0]=(gu-(127-13)*8)[ui.u[0]];ui.l[1]=(gu-(127-13)*8)[ui.u[1
]];ui.l[2]=(gu-(127-13)*8)[ui.u[2]];ui.l[3]=(gu-(127-13)*8)[ui.u[3]];tx=uf.ab;
ty=ug.ab;tz=uh.ab;ua=ui.ab;};{__m128i uj;uj=_mm_srli_epi32(_mm_castps_si128(tt)
,12);(uj)=_mm_and_si128(uj,(hv));(uj)=_mm_or_si128(uj,(hw));(tx)=_mm_madd_epi16
(tx,uj);tx=_mm_srli_epi32(tx,16);};{__m128i uk;uk=_mm_srli_epi32(//////////////
_mm_castps_si128(tu),12);(uk)=_mm_and_si128(uk,(hv));(uk)=_mm_or_si128(uk,(hw))
;(ty)=_mm_madd_epi16(ty,uk);ty=_mm_srli_epi32(ty,16);};{__m128i ul;ul=/////////
_mm_srli_epi32(_mm_castps_si128(tv),12);(ul)=_mm_and_si128(ul,(hv));(ul)=//////
_mm_or_si128(ul,(hw));(tz)=_mm_madd_epi16(tz,ul);tz=_mm_srli_epi32(tz,16);};{//
__m128i um;um=_mm_srli_epi32(_mm_castps_si128(tw),12);(um)=_mm_and_si128(um,(hv
));(um)=_mm_or_si128(um,(hw));(ua)=_mm_madd_epi16(ua,um);ua=_mm_srli_epi32(ua,
16);};tz=_mm_packs_epi32(tz,ty);tx=_mm_packs_epi32(tx,ua);ty=_mm_unpacklo_epi16
(tz,tx);ua=_mm_unpackhi_epi16(tz,tx);tz=_mm_unpacklo_epi16(ty,ua);tx=//////////
_mm_unpackhi_epi16(ty,ua);tz=_mm_packus_epi16(tz,tx);_mm_storeu_si128((__m128i*
)(tq),tz);;tp+=16;tq+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){/////
__asm__(""::"r"(tp));tq[0-4]=gv(tp[2]);tq[1-4]=gv(tp[1]);tq[2-4]=gv(tp[0]);tq[3
-4]=gv(tp[3]);tq+=4;tp+=4;}tq-=4;}static float*kf(float*tn,int to,void const*tp
){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*ts=(/////
unsigned char const*)tp;do{tq[0]=gs[ts[2]];tq[1]=gs[ts[1]];tq[2]=gs[ts[0]];tq[3
]=((float)ts[3])*3.9215689e-03f;ts+=4;tq+=4;}while(tq<tr);return tr;}static////
void kg(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(unsigned
char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=16){float const*ts=tp+
to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp)
);(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=_mm_loadu_ps((float const*)((tp)
+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((float const*)
((tp)+12));{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,
tw);uc=_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud)
;tu=_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)
=_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)
));tx=_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,/////////////
_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=/////////
_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((
ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(////////////
_mm_castps_si128(tv),20);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=//////
_mm_max_ps(tw,_mm_setzero_ps());(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(
tw);;{hk uf,ug,uh;uf.ab=tx;ug.ab=ty;uh.ab=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];
uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(
127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug
.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0
]=(gu-(127-13)*8)[uh.u[0]];uh.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13
)*8)[uh.u[2]];uh.l[3]=(gu-(127-13)*8)[uh.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;};{//
__m128i ui;ui=_mm_srli_epi32(_mm_castps_si128(tt),12);(ui)=_mm_and_si128(ui,(hv
));(ui)=_mm_or_si128(ui,(hw));(tx)=_mm_madd_epi16(tx,ui);tx=_mm_srli_epi32(tx,
16);};{__m128i uj;uj=_mm_srli_epi32(_mm_castps_si128(tu),12);(uj)=_mm_and_si128
(uj,(hv));(uj)=_mm_or_si128(uj,(hw));(ty)=_mm_madd_epi16(ty,uj);ty=////////////
_mm_srli_epi32(ty,16);};{__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tv),12);
(uk)=_mm_and_si128(uk,(hv));(uk)=_mm_or_si128(uk,(hw));(tz)=_mm_madd_epi16(tz,
uk);tz=_mm_srli_epi32(tz,16);};tz=_mm_packs_epi32(tz,ty);tx=_mm_packs_epi32(tx,
ua);ty=_mm_unpacklo_epi16(tz,tx);ua=_mm_unpackhi_epi16(tz,tx);tz=//////////////
_mm_unpacklo_epi16(ty,ua);tx=_mm_unpackhi_epi16(ty,ua);tz=_mm_packus_epi16(tz,
tx);_mm_storeu_si128((__m128i*)(tq),tz);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq
==(tr+16))break;tq=tr;tp=ts;}return;}_Pragma("GCC unroll 1")_Pragma(///////////
"GCC novector")do{float ul;__asm__(""::"r"(tp));tq[2]=gv(tp[0]);tq[1]=gv(tp[1])
;tq[0]=gv(tp[2]);ul=tp[3]*255.0f+0.5f;for(;;){if((ul)<(0))(ul)=(0);if((ul)>(255
))(ul)=(255);break;};tq[3]=(unsigned char)ul;tq+=4;tp+=4;}while(tq<tr);}static
float*kh(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(////
float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;unsigned short//
const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128 tx,ty;__asm__(
""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i tz=///////////
_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz);}
;(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_mul_ps(tx,(hp));(ty
)=_mm_mul_ps(ty,(hp));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tx),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ty),(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq+0)
,tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr
+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=((float)(ts[2]))*////
1.5259022e-05f;tq[1-4]=((float)(ts[1]))*1.5259022e-05f;tq[2-4]=((float)(ts[0]))
*1.5259022e-05f;tq[3-4]=((float)(ts[3]))*1.5259022e-05f;tq+=4;ts+=4;}tq-=4;////
return tr;}static void ki(void*tn,int to,float const*tp){unsigned short*///////
__restrict__ tq=(unsigned short*)tn;unsigned short*tr=((unsigned short*)tq)+to;
{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;
__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((hn),_mm_loadu_ps((float//
const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hn),_mm_loadu_ps((float const*)
(tp+4))));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(2<<0)+(
1<<2)+(0<<4)+(3<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tu),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128i tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps
(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=_mm_cvttps_epi32(_mm_max_ps(////////
_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32
(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};_mm_storeu_si128((
__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=
tr;tp=ts;}return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq
<=tr){__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*
)(tp));(ty)=_mm_add_ps((hq),_mm_mul_ps((hn),ty));(ty)=_mm_castsi128_ps(////////
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128i//
ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=/////////
_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=/////
_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq
-=4;}static float*kj(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;//////
unsigned short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128
tx,ty;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i
tz=_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz
);};(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(ty)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;
ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;//
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[2]));tq[1-4]=((float)(ts[1]));tq[2-4]=((float)(ts[0]));tq
[3-4]=((float)(ts[3]));tq+=4;ts+=4;}tq-=4;return tr;}static void kk(void*tn,int
to,float const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned
short*tr=((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2
;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),////
_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float/////
const*)(tp+4)));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(2
<<0)+(1<<2)+(0<<4)+(3<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////
_mm_castps_si128(tu),(2<<0)+(1<<2)+(0<<4)+(3<<6)));{__m128i tw,tx;tw=//////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=/////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=/////////
_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=/////
_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(
tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}return;}}tq+=4;_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__m128 ty;__m128i tz;//////
__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp));(ty)=_mm_add_ps((hq)
,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(1<<2)
+(0<<4)+(3<<6)));{__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(
hn)),_mm_setzero_ps()));ub=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),////
_mm_setzero_ps()));ua=_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=/////////
_mm_packs_epi32(ua,ub);tz=_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq
-4),(tz));tq+=4;tp+=4;}tq-=4;}static float*kl(float*tn,int to,void const*tp){//
float*__restrict__ tq=tn;float*tr=(float*)tq+to;he const*ts=(he const*)tp;if(to
>=8){he const*tt=ts+to-8;tr-=8;for(;;){__asm__(""::"r"(tq));hh(tq,ts);{__m128//
tu,tv;(tu)=_mm_loadu_ps((float const*)(tq));(tv)=_mm_loadu_ps((float const*)(tq
+4));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(2<<0)+(1<<2)
+(0<<4)+(3<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(
2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq),tu);_mm_storeu_ps((////
float*)(tq+4),tv);}tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=
tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=
tr){__asm__(""::"r"(tq));tq[0-4]=hf(ts[2]);tq[1-4]=hf(ts[1]);tq[2-4]=hf(ts[0]);
tq[3-4]=hf(ts[3]);tq+=4;ts+=4;}tq-=4;return tr;}static void km(void*tn,int to,
float const*tp){he*__restrict__ tq=(he*)tn;he*tr=((he*)tq)+to;if(to>=8){float//
const*ts=tp+to-8;tr-=8;for(;;){__asm__(""::"r"(tp));{__m128 tt[2];(tt[0])=/////
_mm_loadu_ps((float const*)(tp));(tt[1])=_mm_loadu_ps((float const*)(tp+4));(tt
[0])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[0]),(2<<0)+(1<<2)+(
0<<4)+(3<<6)));(tt[1])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[1
]),(2<<0)+(1<<2)+(0<<4)+(3<<6)));hj(tq,(float*)tt);}tp+=8;tq+=8;if(tq<=tr)/////
continue;if(tq==(tr+8))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=hg(tp[2]);tq[
1-4]=hg(tp[1]);tq[2-4]=hg(tp[0]);tq[3-4]=hg(tp[3]);tq+=4;tp+=4;}tq-=4;}static//
float*kn(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(////
float*)tq+to;float const*ts=(float const*)tp;if(to>=16){float const*tt=ts+to-16
;tr-=16;for(;;){__asm__(""::"r"(tq));{__m128 tu,tv,tw,tx;(tu)=_mm_loadu_ps((///
float const*)(ts));(tv)=_mm_loadu_ps((float const*)(ts+4));(tw)=_mm_loadu_ps((
float const*)(ts+8));(tx)=_mm_loadu_ps((float const*)(ts+12));(tu)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(2<<0)+(1<<
2)+(0<<4)+(3<<6)));(tw)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tw)
,(2<<0)+(1<<2)+(0<<4)+(3<<6)));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(////////
_mm_castps_si128(tx),(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq),
tu);_mm_storeu_ps((float*)(tq+4),tv);_mm_storeu_ps((float*)(tq+8),tw);/////////
_mm_storeu_ps((float*)(tq+12),tx);}tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=ts[2];tq[1-4]=ts[1];
tq[2-4]=ts[0];tq[3-4]=ts[3];tq+=4;ts+=4;}tq-=4;return tr;}static void ko(void*
tn,int to,float const*tp){float*__restrict__ tq=(float*)tn;float*tr=((float*)tq
)+to;if(to>=(4*2)){float const*ts=tp+to-(4*2);tr-=(4*2);for(;;){__m128 tt,tu;//
__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)(tp));(tu)=_mm_loadu_ps((
float const*)(tp+4));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tt),(2<<0)+(1<<2)+(0<<4)+(3<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tu),(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq),
tt);_mm_storeu_ps((float*)(tq+4),tu);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==
(tr+(4*2)))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 tv;__asm__(""::"r"(tp));(tv)=_mm_loadu_ps((
float const*)(tp));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv)
,(2<<0)+(1<<2)+(0<<4)+(3<<6)));_mm_storeu_ps((float*)(tq-4),tv);tq+=4;tp+=4;}tq
-=4;}static float*kp(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned
char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128
tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{//
__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8
(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=//////////
_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(
tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);
(tz)=_mm_mul_ps(tz,(ho));(ua)=_mm_mul_ps(ua,(ho));(ub)=_mm_mul_ps(ub,(ho));(uc)
=_mm_mul_ps(uc,(ho));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tz),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ua),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ub)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(uc)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)(tq+4),ua);//////
_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=
16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;///
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[1]))*3.9215689e-03f;tq[1-4]=((float)(ts[2]))*////////////
3.9215689e-03f;tq[2-4]=((float)(ts[3]))*3.9215689e-03f;tq[3-4]=((float)(ts[0]))
*3.9215689e-03f;tq+=4;ts+=4;}tq-=4;return tr;}static void kq(void*tn,int to,///
float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*
tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;)
{__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((
hm),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hm),///
_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(tt),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(tu)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(0<<2)+(1<<4)+(2<<6)));{__m128 tw
,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,//
_mm_setzero_ps());tx=_mm_max_ps(tx,_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz
=_mm_cvttps_epi32(tx);ty=_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};//
_mm_storel_epi64((__m128i*)(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq
==(tr+4*2))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=///
_mm_loadu_ps((float const*)(tp));(ua)=_mm_add_ps((hq),_mm_mul_ps((hm),ua));(ua)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(0<<2)+(1<<4)+(
2<<6)));{__m128 uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);
uc=_mm_max_ps(uc,_mm_setzero_ps());ud=_mm_max_ps(ud,_mm_setzero_ps());ue=//////
_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud);ue=_mm_packs_epi32(ue,uf);ub=/////
_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=
4;}static float*kr(float*tn,int to,void const*tp){float*__restrict__ tq=tn;////
float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned
char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128
tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{//
__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8
(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=//////////
_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(
tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);
(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3<<
4)+(0<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)
+(2<<2)+(3<<4)+(0<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(ub),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(uc)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(2<<2)+(3<<4)+(0<<6)));//////////
_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)(tq+4),ua);_mm_storeu_ps
((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=16;if(tq<=tr)
continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4
]=((float)(ts[1]));tq[1-4]=((float)(ts[2]));tq[2-4]=((float)(ts[3]));tq[3-4]=((
float)(ts[0]));tq+=4;ts+=4;}tq-=4;return tr;}static void ks(void*tn,int to,////
float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*
tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;)
{__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_loadu_ps
((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp+4)));
(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(3<<0)+(0<<2)+(1<<
4)+(2<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)
+(0<<2)+(1<<4)+(2<<6)));{__m128 tw,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=///
_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,_mm_setzero_ps());tx=_mm_max_ps(tx,/////////
_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx);ty=//////////
_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};_mm_storel_epi64((__m128i*)
(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts
;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){///
__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*)(tp))
;(ua)=_mm_add_ps((hq),ua);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(ua),(3<<0)+(0<<2)+(1<<4)+(2<<6)));{__m128 uc,ud;__m128i ue,uf;
uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,_mm_setzero_ps());ud
=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud
);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=////////
_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;}static float*kt(float*tn,int to,void
const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*
ts=(unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[1]];tq[1-4]=gs[ts
[2]];tq[2-4]=gs[ts[3]];tq[3-4]=gs[ts[0]];tq+=4;ts+=4;}tq-=4;return tr;}static//
void ku(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(unsigned
char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=16){float const*ts=tp+
to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp)
);(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=_mm_loadu_ps((float const*)((tp)
+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((float const*)
((tp)+12));{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,
tw);uc=_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud)
;tu=_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)
=_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)
));tx=_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,/////////////
_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=/////////
_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((
ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(////////////
_mm_castps_si128(tv),20);;(tw)=_mm_max_ps(tw,_mm_castsi128_ps((ht)));(tw)=/////
_mm_min_ps(tw,_mm_castsi128_ps((hu)));ua=_mm_srli_epi32(_mm_castps_si128(tw),20
);;{hk uf,ug,uh,ui;uf.ab=tx;ug.ab=ty;uh.ab=tz;ui.ab=ua;uf.l[0]=(gu-(127-13)*8)[
uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l
[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-
13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3
]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(
gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(127-13)*8)[uh.u[3]];ui.l[0]=(gu-(127-13)*8
)[ui.u[0]];ui.l[1]=(gu-(127-13)*8)[ui.u[1]];ui.l[2]=(gu-(127-13)*8)[ui.u[2]];ui
.l[3]=(gu-(127-13)*8)[ui.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;ua=ui.ab;};{__m128i//
uj;uj=_mm_srli_epi32(_mm_castps_si128(tt),12);(uj)=_mm_and_si128(uj,(hv));(uj)=
_mm_or_si128(uj,(hw));(tx)=_mm_madd_epi16(tx,uj);tx=_mm_srli_epi32(tx,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tu),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(ty)=_mm_madd_epi16(ty,uk);ty=_mm_srli_epi32(ty,
16);};{__m128i ul;ul=_mm_srli_epi32(_mm_castps_si128(tv),12);(ul)=_mm_and_si128
(ul,(hv));(ul)=_mm_or_si128(ul,(hw));(tz)=_mm_madd_epi16(tz,ul);tz=////////////
_mm_srli_epi32(tz,16);};{__m128i um;um=_mm_srli_epi32(_mm_castps_si128(tw),12);
(um)=_mm_and_si128(um,(hv));(um)=_mm_or_si128(um,(hw));(ua)=_mm_madd_epi16(ua,
um);ua=_mm_srli_epi32(ua,16);};ua=_mm_packs_epi32(ua,tx);ty=_mm_packs_epi32(ty,
tz);tx=_mm_unpacklo_epi16(ua,ty);tz=_mm_unpackhi_epi16(ua,ty);ua=//////////////
_mm_unpacklo_epi16(tx,tz);ty=_mm_unpackhi_epi16(tx,tz);ua=_mm_packus_epi16(ua,
ty);_mm_storeu_si128((__m128i*)(tq),ua);;tp+=16;tq+=16;if(tq<=tr)continue;if(tq
==(tr+16))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tp));tq[0-4]=gv(tp[3]);tq[1-4]=gv(
tp[0]);tq[2-4]=gv(tp[1]);tq[3-4]=gv(tp[2]);tq+=4;tp+=4;}tq-=4;}static float*kv(
float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;
unsigned char const*ts=(unsigned char const*)tp;do{tq[0]=gs[ts[1]];tq[1]=gs[ts[
2]];tq[2]=gs[ts[3]];tq[3]=((float)ts[0])*3.9215689e-03f;ts+=4;tq+=4;}while(tq<
tr);return tr;}static void kw(void*tn,int to,float const*tp){unsigned char*////
__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(
to>=16){float const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty
,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=//////
_mm_loadu_ps((float const*)((tp)+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));
(tw)=_mm_loadu_ps((float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=/////////////
_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=
_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=/////
_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,/////////////
_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=/////////
_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,_mm_castsi128_ps((
ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=_mm_srli_epi32(////////////
_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=/////
_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(_mm_castps_si128(tv),20
);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=_mm_max_ps(tw,_mm_setzero_ps(
));(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(tw);;{hk uf,ug,uh;uf.ab=tx;ug
.ab=ty;uh.ab=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1
]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(
gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8
)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh
.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(//
127-13)*8)[uh.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;};{__m128i ui;ui=_mm_srli_epi32(
_mm_castps_si128(tt),12);(ui)=_mm_and_si128(ui,(hv));(ui)=_mm_or_si128(ui,(hw))
;(tx)=_mm_madd_epi16(tx,ui);tx=_mm_srli_epi32(tx,16);};{__m128i uj;uj=/////////
_mm_srli_epi32(_mm_castps_si128(tu),12);(uj)=_mm_and_si128(uj,(hv));(uj)=//////
_mm_or_si128(uj,(hw));(ty)=_mm_madd_epi16(ty,uj);ty=_mm_srli_epi32(ty,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tv),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(tz)=_mm_madd_epi16(tz,uk);tz=_mm_srli_epi32(tz,
16);};ua=_mm_packs_epi32(ua,tx);ty=_mm_packs_epi32(ty,tz);tx=_mm_unpacklo_epi16
(ua,ty);tz=_mm_unpackhi_epi16(ua,ty);ua=_mm_unpacklo_epi16(tx,tz);ty=//////////
_mm_unpackhi_epi16(tx,tz);ua=_mm_packus_epi16(ua,ty);_mm_storeu_si128((__m128i*
)(tq),ua);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float ul;__asm__(""::
"r"(tp));tq[1]=gv(tp[0]);tq[2]=gv(tp[1]);tq[3]=gv(tp[2]);ul=tp[3]*255.0f+0.5f;
for(;;){if((ul)<(0))(ul)=(0);if((ul)>(255))(ul)=(255);break;};tq[0]=(unsigned//
char)ul;tq+=4;tp+=4;}while(tq<tr);}static float*kx(float*tn,int to,void const*
tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned short const*ts=(//
unsigned short const*)tp;unsigned short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;
){__m128i tu,tv,tw;__m128 tx,ty;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((////
__m128i const*)(ts));{__m128i tz=_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,
tz);tw=_mm_unpackhi_epi16(tu,tz);};(tx)=_mm_cvtepi32_ps(tv);(ty)=//////////////
_mm_cvtepi32_ps(tw);(tx)=_mm_mul_ps(tx,(hp));(ty)=_mm_mul_ps(ty,(hp));(tx)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(1<<0)+(2<<
2)+(3<<4)+(0<<6)));_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4
),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr
+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(
""::"r"(tq));tq[0-4]=((float)(ts[1]))*1.5259022e-05f;tq[1-4]=((float)(ts[2]))*
1.5259022e-05f;tq[2-4]=((float)(ts[3]))*1.5259022e-05f;tq[3-4]=((float)(ts[0]))
*1.5259022e-05f;tq+=4;ts+=4;}tq-=4;return tr;}static void ky(void*tn,int to,///
float const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned////
short*tr=((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2
;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),////
_mm_mul_ps((hn),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),///////
_mm_mul_ps((hn),_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(////
_mm_shuffle_epi32(_mm_castps_si128(tt),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(tu)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(0<<2)+(1<<4)+(2
<<6)));{__m128i tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),//////
_mm_setzero_ps()));tx=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),/////////
_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=/////////
_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq
),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}///
return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){////
__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp))
;(ty)=_mm_add_ps((hq),_mm_mul_ps((hn),ty));(ty)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(0<<2)+(1<<4)+(2<<6)));{__m128i//
ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=/////////
_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=/////
_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq
-=4;}static float*kz(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;//////
unsigned short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128
tx,ty;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i
tz=_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz
);};(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;
ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;//
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[1]));tq[1-4]=((float)(ts[2]));tq[2-4]=((float)(ts[3]));tq
[3-4]=((float)(ts[0]));tq+=4;ts+=4;}tq-=4;return tr;}static void la(void*tn,int
to,float const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned
short*tr=((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2
;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),////
_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float/////
const*)(tp+4)));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(3
<<0)+(0<<2)+(1<<4)+(2<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////
_mm_castps_si128(tu),(3<<0)+(0<<2)+(1<<4)+(2<<6)));{__m128i tw,tx;tw=//////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=/////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=/////////
_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=/////
_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(
tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}return;}}tq+=4;_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__m128 ty;__m128i tz;//////
__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp));(ty)=_mm_add_ps((hq)
,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(0<<2)
+(1<<4)+(2<<6)));{__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(
hn)),_mm_setzero_ps()));ub=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),////
_mm_setzero_ps()));ua=_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=/////////
_mm_packs_epi32(ua,ub);tz=_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq
-4),(tz));tq+=4;tp+=4;}tq-=4;}static float*lb(float*tn,int to,void const*tp){//
float*__restrict__ tq=tn;float*tr=(float*)tq+to;he const*ts=(he const*)tp;if(to
>=8){he const*tt=ts+to-8;tr-=8;for(;;){__asm__(""::"r"(tq));hh(tq,ts);{__m128//
tu,tv;(tu)=_mm_loadu_ps((float const*)(tq));(tv)=_mm_loadu_ps((float const*)(tq
+4));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(2<<2)
+(3<<4)+(0<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(
1<<0)+(2<<2)+(3<<4)+(0<<6)));_mm_storeu_ps((float*)(tq),tu);_mm_storeu_ps((////
float*)(tq+4),tv);}tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=
tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=
tr){__asm__(""::"r"(tq));tq[0-4]=hf(ts[1]);tq[1-4]=hf(ts[2]);tq[2-4]=hf(ts[3]);
tq[3-4]=hf(ts[0]);tq+=4;ts+=4;}tq-=4;return tr;}static void lc(void*tn,int to,
float const*tp){he*__restrict__ tq=(he*)tn;he*tr=((he*)tq)+to;if(to>=8){float//
const*ts=tp+to-8;tr-=8;for(;;){__asm__(""::"r"(tp));{__m128 tt[2];(tt[0])=/////
_mm_loadu_ps((float const*)(tp));(tt[1])=_mm_loadu_ps((float const*)(tp+4));(tt
[0])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[0]),(3<<0)+(0<<2)+(
1<<4)+(2<<6)));(tt[1])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[1
]),(3<<0)+(0<<2)+(1<<4)+(2<<6)));hj(tq,(float*)tt);}tp+=8;tq+=8;if(tq<=tr)/////
continue;if(tq==(tr+8))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=hg(tp[3]);tq[
1-4]=hg(tp[0]);tq[2-4]=hg(tp[1]);tq[3-4]=hg(tp[2]);tq+=4;tp+=4;}tq-=4;}static//
float*ld(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(////
float*)tq+to;float const*ts=(float const*)tp;if(to>=16){float const*tt=ts+to-16
;tr-=16;for(;;){__asm__(""::"r"(tq));{__m128 tu,tv,tw,tx;(tu)=_mm_loadu_ps((///
float const*)(ts));(tv)=_mm_loadu_ps((float const*)(ts+4));(tw)=_mm_loadu_ps((
float const*)(ts+8));(tx)=_mm_loadu_ps((float const*)(ts+12));(tu)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(1<<0)+(2<<
2)+(3<<4)+(0<<6)));(tw)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tw)
,(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(////////
_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));_mm_storeu_ps((float*)(tq),
tu);_mm_storeu_ps((float*)(tq+4),tv);_mm_storeu_ps((float*)(tq+8),tw);/////////
_mm_storeu_ps((float*)(tq+12),tx);}tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=ts[1];tq[1-4]=ts[2];
tq[2-4]=ts[3];tq[3-4]=ts[0];tq+=4;ts+=4;}tq-=4;return tr;}static void le(void*
tn,int to,float const*tp){float*__restrict__ tq=(float*)tn;float*tr=((float*)tq
)+to;if(to>=(4*2)){float const*ts=tp+to-(4*2);tr-=(4*2);for(;;){__m128 tt,tu;//
__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)(tp));(tu)=_mm_loadu_ps((
float const*)(tp+4));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tt),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tu),(3<<0)+(0<<2)+(1<<4)+(2<<6)));_mm_storeu_ps((float*)(tq),
tt);_mm_storeu_ps((float*)(tq+4),tu);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==
(tr+(4*2)))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 tv;__asm__(""::"r"(tp));(tv)=_mm_loadu_ps((
float const*)(tp));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv)
,(3<<0)+(0<<2)+(1<<4)+(2<<6)));_mm_storeu_ps((float*)(tq-4),tv);tq+=4;tp+=4;}tq
-=4;}static float*lf(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned
char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128
tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{//
__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8
(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=//////////
_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(
tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);
(tz)=_mm_mul_ps(tz,(ho));(ua)=_mm_mul_ps(ua,(ho));(ub)=_mm_mul_ps(ub,(ho));(uc)
=_mm_mul_ps(uc,(ho));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tz),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ua),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(ub)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(uc)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(2<<2)+(1<<4)+(0
<<6)));_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)(tq+4),ua);//////
_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=
16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;///
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[3]))*3.9215689e-03f;tq[1-4]=((float)(ts[2]))*////////////
3.9215689e-03f;tq[2-4]=((float)(ts[1]))*3.9215689e-03f;tq[3-4]=((float)(ts[0]))
*3.9215689e-03f;tq+=4;ts+=4;}tq-=4;return tr;}static void lg(void*tn,int to,///
float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*
tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;)
{__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((
hm),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hm),///
_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(tt),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(tu)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(2<<2)+(1<<4)+(0<<6)));{__m128 tw
,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,//
_mm_setzero_ps());tx=_mm_max_ps(tx,_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz
=_mm_cvttps_epi32(tx);ty=_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};//
_mm_storel_epi64((__m128i*)(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq
==(tr+4*2))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=///
_mm_loadu_ps((float const*)(tp));(ua)=_mm_add_ps((hq),_mm_mul_ps((hm),ua));(ua)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(2<<2)+(1<<4)+(
0<<6)));{__m128 uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);
uc=_mm_max_ps(uc,_mm_setzero_ps());ud=_mm_max_ps(ud,_mm_setzero_ps());ue=//////
_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud);ue=_mm_packs_epi32(ue,uf);ub=/////
_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=
4;}static float*lh(float*tn,int to,void const*tp){float*__restrict__ tq=tn;////
float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned
char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128
tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{//
__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8
(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=//////////
_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(
tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);
(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(3<<0)+(2<<2)+(1<<
4)+(0<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)
+(2<<2)+(1<<4)+(0<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(ub),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(uc)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(2<<2)+(1<<4)+(0<<6)));//////////
_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)(tq+4),ua);_mm_storeu_ps
((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=16;if(tq<=tr)
continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4
]=((float)(ts[3]));tq[1-4]=((float)(ts[2]));tq[2-4]=((float)(ts[1]));tq[3-4]=((
float)(ts[0]));tq+=4;ts+=4;}tq-=4;return tr;}static void li(void*tn,int to,////
float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*
tr=((unsigned char*)tq)+to;if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;)
{__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_loadu_ps
((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp+4)));
(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(3<<0)+(2<<2)+(1<<
4)+(0<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)
+(2<<2)+(1<<4)+(0<<6)));{__m128 tw,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=///
_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,_mm_setzero_ps());tx=_mm_max_ps(tx,/////////
_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx);ty=//////////
_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};_mm_storel_epi64((__m128i*)
(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts
;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){///
__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*)(tp))
;(ua)=_mm_add_ps((hq),ua);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(ua),(3<<0)+(2<<2)+(1<<4)+(0<<6)));{__m128 uc,ud;__m128i ue,uf;
uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,_mm_setzero_ps());ud
=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf=_mm_cvttps_epi32(ud
);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(int*)(tq-4)=////////
_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;}static float*lj(float*tn,int to,void
const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*
ts=(unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[3]];tq[1-4]=gs[ts
[2]];tq[2-4]=gs[ts[1]];tq[3-4]=gs[ts[0]];tq+=4;ts+=4;}tq-=4;return tr;}static//
void lk(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(unsigned
char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=16){float const*ts=tp+
to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp)
);(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=_mm_loadu_ps((float const*)((tp)
+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((float const*)
((tp)+12));{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,
tw);uc=_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud)
;tu=_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)
=_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)
));tx=_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,/////////////
_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=/////////
_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((
ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(////////////
_mm_castps_si128(tv),20);;(tw)=_mm_max_ps(tw,_mm_castsi128_ps((ht)));(tw)=/////
_mm_min_ps(tw,_mm_castsi128_ps((hu)));ua=_mm_srli_epi32(_mm_castps_si128(tw),20
);;{hk uf,ug,uh,ui;uf.ab=tx;ug.ab=ty;uh.ab=tz;ui.ab=ua;uf.l[0]=(gu-(127-13)*8)[
uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l
[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-
13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3
]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(
gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(127-13)*8)[uh.u[3]];ui.l[0]=(gu-(127-13)*8
)[ui.u[0]];ui.l[1]=(gu-(127-13)*8)[ui.u[1]];ui.l[2]=(gu-(127-13)*8)[ui.u[2]];ui
.l[3]=(gu-(127-13)*8)[ui.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;ua=ui.ab;};{__m128i//
uj;uj=_mm_srli_epi32(_mm_castps_si128(tt),12);(uj)=_mm_and_si128(uj,(hv));(uj)=
_mm_or_si128(uj,(hw));(tx)=_mm_madd_epi16(tx,uj);tx=_mm_srli_epi32(tx,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tu),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(ty)=_mm_madd_epi16(ty,uk);ty=_mm_srli_epi32(ty,
16);};{__m128i ul;ul=_mm_srli_epi32(_mm_castps_si128(tv),12);(ul)=_mm_and_si128
(ul,(hv));(ul)=_mm_or_si128(ul,(hw));(tz)=_mm_madd_epi16(tz,ul);tz=////////////
_mm_srli_epi32(tz,16);};{__m128i um;um=_mm_srli_epi32(_mm_castps_si128(tw),12);
(um)=_mm_and_si128(um,(hv));(um)=_mm_or_si128(um,(hw));(ua)=_mm_madd_epi16(ua,
um);ua=_mm_srli_epi32(ua,16);};ua=_mm_packs_epi32(ua,tz);ty=_mm_packs_epi32(ty,
tx);tz=_mm_unpacklo_epi16(ua,ty);tx=_mm_unpackhi_epi16(ua,ty);ua=//////////////
_mm_unpacklo_epi16(tz,tx);ty=_mm_unpackhi_epi16(tz,tx);ua=_mm_packus_epi16(ua,
ty);_mm_storeu_si128((__m128i*)(tq),ua);;tp+=16;tq+=16;if(tq<=tr)continue;if(tq
==(tr+16))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tp));tq[0-4]=gv(tp[3]);tq[1-4]=gv(
tp[2]);tq[2-4]=gv(tp[1]);tq[3-4]=gv(tp[0]);tq+=4;tp+=4;}tq-=4;}static float*ll(
float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;
unsigned char const*ts=(unsigned char const*)tp;do{tq[0]=gs[ts[3]];tq[1]=gs[ts[
2]];tq[2]=gs[ts[1]];tq[3]=((float)ts[0])*3.9215689e-03f;ts+=4;tq+=4;}while(tq<
tr);return tr;}static void lm(void*tn,int to,float const*tp){unsigned char*////
__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(
to>=16){float const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,tw;__m128i tx,ty
,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((tp)));(tu)=//////
_mm_loadu_ps((float const*)((tp)+4));(tv)=_mm_loadu_ps((float const*)((tp)+8));
(tw)=_mm_loadu_ps((float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=/////////////
_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=
_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=/////
_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,/////////////
_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=/////////
_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,_mm_castsi128_ps((
ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=_mm_srli_epi32(////////////
_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=/////
_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(_mm_castps_si128(tv),20
);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=_mm_max_ps(tw,_mm_setzero_ps(
));(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(tw);;{hk uf,ug,uh;uf.ab=tx;ug
.ab=ty;uh.ab=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1
]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(
gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8
)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh
.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(//
127-13)*8)[uh.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;};{__m128i ui;ui=_mm_srli_epi32(
_mm_castps_si128(tt),12);(ui)=_mm_and_si128(ui,(hv));(ui)=_mm_or_si128(ui,(hw))
;(tx)=_mm_madd_epi16(tx,ui);tx=_mm_srli_epi32(tx,16);};{__m128i uj;uj=/////////
_mm_srli_epi32(_mm_castps_si128(tu),12);(uj)=_mm_and_si128(uj,(hv));(uj)=//////
_mm_or_si128(uj,(hw));(ty)=_mm_madd_epi16(ty,uj);ty=_mm_srli_epi32(ty,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tv),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(tz)=_mm_madd_epi16(tz,uk);tz=_mm_srli_epi32(tz,
16);};ua=_mm_packs_epi32(ua,tz);ty=_mm_packs_epi32(ty,tx);tz=_mm_unpacklo_epi16
(ua,ty);tx=_mm_unpackhi_epi16(ua,ty);ua=_mm_unpacklo_epi16(tz,tx);ty=//////////
_mm_unpackhi_epi16(tz,tx);ua=_mm_packus_epi16(ua,ty);_mm_storeu_si128((__m128i*
)(tq),ua);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float ul;__asm__(""::
"r"(tp));tq[3]=gv(tp[0]);tq[2]=gv(tp[1]);tq[1]=gv(tp[2]);ul=tp[3]*255.0f+0.5f;
for(;;){if((ul)<(0))(ul)=(0);if((ul)>(255))(ul)=(255);break;};tq[0]=(unsigned//
char)ul;tq+=4;tp+=4;}while(tq<tr);}static float*ln(float*tn,int to,void const*
tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned short const*ts=(//
unsigned short const*)tp;unsigned short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;
){__m128i tu,tv,tw;__m128 tx,ty;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((////
__m128i const*)(ts));{__m128i tz=_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,
tz);tw=_mm_unpackhi_epi16(tu,tz);};(tx)=_mm_cvtepi32_ps(tv);(ty)=//////////////
_mm_cvtepi32_ps(tw);(tx)=_mm_mul_ps(tx,(hp));(ty)=_mm_mul_ps(ty,(hp));(tx)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(3<<0)+(2<<2)+(1<<4)+(0
<<6)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(2<<
2)+(1<<4)+(0<<6)));_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4
),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr
+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(
""::"r"(tq));tq[0-4]=((float)(ts[3]))*1.5259022e-05f;tq[1-4]=((float)(ts[2]))*
1.5259022e-05f;tq[2-4]=((float)(ts[1]))*1.5259022e-05f;tq[3-4]=((float)(ts[0]))
*1.5259022e-05f;tq+=4;ts+=4;}tq-=4;return tr;}static void lp(void*tn,int to,///
float const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned////
short*tr=((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2
;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),////
_mm_mul_ps((hn),_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),///////
_mm_mul_ps((hn),_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(////
_mm_shuffle_epi32(_mm_castps_si128(tt),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(tu)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(2<<2)+(1<<4)+(0
<<6)));{__m128i tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),//////
_mm_setzero_ps()));tx=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),/////////
_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=/////////
_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq
),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}///
return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){////
__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp))
;(ty)=_mm_add_ps((hq),_mm_mul_ps((hn),ty));(ty)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(2<<2)+(1<<4)+(0<<6)));{__m128i//
ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=/////////
_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=/////
_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq
-=4;}static float*lq(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;//////
unsigned short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128
tx,ty;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i
tz=_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz
);};(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tx),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(ty)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(2<<2)+(1<<4)+(0
<<6)));_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;
ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;//
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[3]));tq[1-4]=((float)(ts[2]));tq[2-4]=((float)(ts[1]));tq
[3-4]=((float)(ts[0]));tq+=4;ts+=4;}tq-=4;return tr;}static void lr(void*tn,int
to,float const*tp){unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned
short*tr=((unsigned short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2
;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),////
_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float/////
const*)(tp+4)));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(3
<<0)+(2<<2)+(1<<4)+(0<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////
_mm_castps_si128(tu),(3<<0)+(2<<2)+(1<<4)+(0<<6)));{__m128i tw,tx;tw=//////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=/////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=/////////
_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=/////
_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(
tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}return;}}tq+=4;_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__m128 ty;__m128i tz;//////
__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp));(ty)=_mm_add_ps((hq)
,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(2<<2)
+(1<<4)+(0<<6)));{__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(
hn)),_mm_setzero_ps()));ub=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),////
_mm_setzero_ps()));ua=_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=/////////
_mm_packs_epi32(ua,ub);tz=_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq
-4),(tz));tq+=4;tp+=4;}tq-=4;}static float*ls(float*tn,int to,void const*tp){//
float*__restrict__ tq=tn;float*tr=(float*)tq+to;he const*ts=(he const*)tp;if(to
>=8){he const*tt=ts+to-8;tr-=8;for(;;){__asm__(""::"r"(tq));hh(tq,ts);{__m128//
tu,tv;(tu)=_mm_loadu_ps((float const*)(tq));(tv)=_mm_loadu_ps((float const*)(tq
+4));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(2<<2)
+(1<<4)+(0<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(
3<<0)+(2<<2)+(1<<4)+(0<<6)));_mm_storeu_ps((float*)(tq),tu);_mm_storeu_ps((////
float*)(tq+4),tv);}tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=
tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=
tr){__asm__(""::"r"(tq));tq[0-4]=hf(ts[3]);tq[1-4]=hf(ts[2]);tq[2-4]=hf(ts[1]);
tq[3-4]=hf(ts[0]);tq+=4;ts+=4;}tq-=4;return tr;}static void lt(void*tn,int to,
float const*tp){he*__restrict__ tq=(he*)tn;he*tr=((he*)tq)+to;if(to>=8){float//
const*ts=tp+to-8;tr-=8;for(;;){__asm__(""::"r"(tp));{__m128 tt[2];(tt[0])=/////
_mm_loadu_ps((float const*)(tp));(tt[1])=_mm_loadu_ps((float const*)(tp+4));(tt
[0])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[0]),(3<<0)+(2<<2)+(
1<<4)+(0<<6)));(tt[1])=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt[1
]),(3<<0)+(2<<2)+(1<<4)+(0<<6)));hj(tq,(float*)tt);}tp+=8;tq+=8;if(tq<=tr)/////
continue;if(tq==(tr+8))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=hg(tp[3]);tq[
1-4]=hg(tp[2]);tq[2-4]=hg(tp[1]);tq[3-4]=hg(tp[0]);tq+=4;tp+=4;}tq-=4;}static//
float*lu(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(////
float*)tq+to;float const*ts=(float const*)tp;if(to>=16){float const*tt=ts+to-16
;tr-=16;for(;;){__asm__(""::"r"(tq));{__m128 tu,tv,tw,tx;(tu)=_mm_loadu_ps((///
float const*)(ts));(tv)=_mm_loadu_ps((float const*)(ts+4));(tw)=_mm_loadu_ps((
float const*)(ts+8));(tx)=_mm_loadu_ps((float const*)(ts+12));(tu)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(2<<2)+(1<<4)+(0
<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(3<<0)+(2<<
2)+(1<<4)+(0<<6)));(tw)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tw)
,(3<<0)+(2<<2)+(1<<4)+(0<<6)));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(////////
_mm_castps_si128(tx),(3<<0)+(2<<2)+(1<<4)+(0<<6)));_mm_storeu_ps((float*)(tq),
tu);_mm_storeu_ps((float*)(tq+4),tv);_mm_storeu_ps((float*)(tq+8),tw);/////////
_mm_storeu_ps((float*)(tq+12),tx);}tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=ts[3];tq[1-4]=ts[2];
tq[2-4]=ts[1];tq[3-4]=ts[0];tq+=4;ts+=4;}tq-=4;return tr;}static void lv(void*
tn,int to,float const*tp){float*__restrict__ tq=(float*)tn;float*tr=((float*)tq
)+to;if(to>=(4*2)){float const*ts=tp+to-(4*2);tr-=(4*2);for(;;){__m128 tt,tu;//
__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)(tp));(tu)=_mm_loadu_ps((
float const*)(tp+4));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tt),(3<<0)+(2<<2)+(1<<4)+(0<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tu),(3<<0)+(2<<2)+(1<<4)+(0<<6)));_mm_storeu_ps((float*)(tq),
tt);_mm_storeu_ps((float*)(tq+4),tu);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==
(tr+(4*2)))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 tv;__asm__(""::"r"(tp));(tv)=_mm_loadu_ps((
float const*)(tp));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv)
,(3<<0)+(2<<2)+(1<<4)+(0<<6)));_mm_storeu_ps((float*)(tq-4),tv);tq+=4;tp+=4;}tq
-=4;}static float*lw(float*tn,int to,void const*tp){float*__restrict__ tq=tn;//
float*tr=(float*)tq+to;unsigned char const*ts=(unsigned char const*)tp;unsigned
char const*tt=ts+to-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128
tz,ua,ub,uc;__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{//
__m128i ud=_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8
(tu,ud);tv=_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=//////////
_mm_unpacklo_epi16(ty,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(
tv);(ua)=_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);
(tz)=_mm_mul_ps(tz,(ho));(ua)=_mm_mul_ps(ua,(ho));(ub)=_mm_mul_ps(ub,(ho));(uc)
=_mm_mul_ps(uc,(ho));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tz),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ua),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(ub)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(uc)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));_mm_storeu_ps((float*)(tq+0),tz);_mm_storeu_ps((float*)(tq+4),ua);//////
_mm_storeu_ps((float*)(tq+8),ub);_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=
16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;///
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=((float)(ts[1]))*3.9215689e-03f;tq[1-4]=((float)(ts[0]))*////////////
3.9215689e-03f;tq[2-4]=((float)(ts[3]))*3.9215689e-03f;tq[3-4]=((float)(ts[2]))
*3.9215689e-03f;tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=((float)(ts[1]))*///////
3.9215689e-03f;tq[1]=((float)(ts[0]))*3.9215689e-03f;tq+=2;ts+=2;}return tr;}//
static void lx(void*tn,int to,float const*tp){unsigned char*__restrict__ tq=(//
unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=4*2){float///
const*ts=tp+to-4*2;tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp))
;(tt)=_mm_add_ps((hq),_mm_mul_ps((hm),_mm_loadu_ps((float const*)(tp))));(tu)=
_mm_add_ps((hq),_mm_mul_ps((hm),_mm_loadu_ps((float const*)(tp+4))));(tt)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<
2)+(3<<4)+(2<<6)));{__m128 tw,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=////////
_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,_mm_setzero_ps());tx=_mm_max_ps(tx,/////////
_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz=_mm_cvttps_epi32(tx);ty=//////////
_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};_mm_storel_epi64((__m128i*)
(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts
;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){///
__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=_mm_loadu_ps((float const*)(tp))
;(ua)=_mm_add_ps((hq),_mm_mul_ps((hm),ua));(ua)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128 uc
,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,//
_mm_setzero_ps());ud=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf
=_mm_cvttps_epi32(ud);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(
int*)(tq-4)=_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")///
_Pragma("GCC novector")while(tq<tr){__m128 ug;__asm__(""::"r"(tp));(ug)=///////
_mm_add_ss((hq),_mm_mul_ss((hm),_mm_load_ss((float const*)(tp+1))));tq[0]=((///
unsigned char)_mm_cvtsi128_si32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ug,(hm))
,_mm_setzero_ps()))));(ug)=_mm_add_ss((hq),_mm_mul_ss((hm),_mm_load_ss((float//
const*)(tp+0))));tq[1]=((unsigned char)_mm_cvtsi128_si32(_mm_cvttps_epi32(/////
_mm_max_ps(_mm_min_ps(ug,(hm)),_mm_setzero_ps()))));tq+=2;tp+=2;}}static float*
ly(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+
to;unsigned char const*ts=(unsigned char const*)tp;unsigned char const*tt=ts+to
-16;if(to>=16){tr-=16;for(;;){__m128i tu,tv,tw,tx,ty;__m128 tz,ua,ub,uc;__asm__
(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i ud=//////////
_mm_setzero_si128();tx=_mm_unpacklo_epi8(tu,ud);ty=_mm_unpackhi_epi8(tu,ud);tv=
_mm_unpacklo_epi16(tx,ud);tw=_mm_unpackhi_epi16(tx,ud);tx=_mm_unpacklo_epi16(ty
,ud);ty=_mm_unpackhi_epi16(ty,ud);};(tz)=_mm_cvtepi32_ps(tv);(ua)=/////////////
_mm_cvtepi32_ps(tw);(ub)=_mm_cvtepi32_ps(tx);(uc)=_mm_cvtepi32_ps(ty);(tz)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(0<<
2)+(3<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub)
,(1<<0)+(0<<2)+(3<<4)+(2<<6)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(////////
_mm_castps_si128(uc),(1<<0)+(0<<2)+(3<<4)+(2<<6)));_mm_storeu_ps((float*)(tq+0)
,tz);_mm_storeu_ps((float*)(tq+4),ua);_mm_storeu_ps((float*)(tq+8),ub);////////
_mm_storeu_ps((float*)(tq+12),uc);tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=((float)(ts[1]));tq[1
-4]=((float)(ts[0]));tq[2-4]=((float)(ts[3]));tq[3-4]=((float)(ts[2]));tq+=4;ts
+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(
""::"r"(tq));tq[0]=((float)(ts[1]));tq[1]=((float)(ts[0]));tq+=2;ts+=2;}return
tr;}static void lz(void*tn,int to,float const*tp){unsigned char*__restrict__ tq
=(unsigned char*)tn;unsigned char*tr=((unsigned char*)tq)+to;if(to>=4*2){float
const*ts=tp+to-4*2;tr-=4*2;for(;;){__m128 tt,tu;__m128i tv;__asm__(""::"r"(tp))
;(tt)=_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp)));(tu)=_mm_add_ps((hq),//
_mm_loadu_ps((float const*)(tp+4)));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(tt),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(tu)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128 tw
,tx;__m128i ty,tz;tw=_mm_min_ps(tt,hm);tx=_mm_min_ps(tu,hm);tw=_mm_max_ps(tw,//
_mm_setzero_ps());tx=_mm_max_ps(tx,_mm_setzero_ps());ty=_mm_cvttps_epi32(tw);tz
=_mm_cvttps_epi32(tx);ty=_mm_packs_epi32(ty,tz);tv=_mm_packus_epi16(ty,ty);};//
_mm_storel_epi64((__m128i*)(tq),(tv));tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq
==(tr+4*2))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tq<=tr){__m128 ua;__m128i ub;__asm__(""::"r"(tp));(ua)=///
_mm_loadu_ps((float const*)(tp));(ua)=_mm_add_ps((hq),ua);(ua)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128//
uc,ud;__m128i ue,uf;uc=_mm_min_ps(ua,hm);ud=_mm_min_ps(ua,hm);uc=_mm_max_ps(uc,
_mm_setzero_ps());ud=_mm_max_ps(ud,_mm_setzero_ps());ue=_mm_cvttps_epi32(uc);uf
=_mm_cvttps_epi32(ud);ue=_mm_packs_epi32(ue,uf);ub=_mm_packus_epi16(ue,ue);};*(
int*)(tq-4)=_mm_cvtsi128_si32(ub);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")///
_Pragma("GCC novector")while(tq<tr){float ug;__asm__(""::"r"(tp));ug=tp[1]+0.5f
;for(;;){if((ug)<(0))(ug)=(0);if((ug)>(255))(ug)=(255);break;};tq[0]=(unsigned
char)ug;ug=tp[0]+0.5f;for(;;){if((ug)<(0))(ug)=(0);if((ug)>(255))(ug)=(255);///
break;};tq[1]=(unsigned char)ug;tq+=2;tp+=2;}}static float*ma(float*tn,int to,
void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char///
const*ts=(unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[1]];tq[1-4]
=gs[ts[0]];tq[2-4]=gs[ts[3]];tq[3-4]=gs[ts[2]];tq+=4;ts+=4;}tq-=4;_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=
gs[ts[1]];tq[1]=gs[ts[0]];tq+=2;ts+=2;}return tr;}static void mb(void*tn,int to
,float const*tp){unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char
*tr=((unsigned char*)tq)+to;if(to>=16){float const*ts=tp+to-16;tr-=16;for(;;){
__m128 tt,tu,tv,tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((
float const*)((tp)));(tu)=_mm_loadu_ps((float const*)((tp)+4));(tv)=///////////
_mm_loadu_ps((float const*)((tp)+8));(tw)=_mm_loadu_ps((float const*)((tp)+12))
;{__m128 ub,uc,ud,ue;ub=_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=///
_mm_unpackhi_ps(tt,tu);ue=_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=///
_mm_movehl_ps(ud,ub);tv=_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=///
_mm_max_ps(tt,_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu))
);tx=_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_max_ps(tu,//////////////
_mm_castsi128_ps((ht)));(tu)=_mm_min_ps(tu,_mm_castsi128_ps((hu)));ty=/////////
_mm_srli_epi32(_mm_castps_si128(tu),20);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((
ht)));(tv)=_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(////////////
_mm_castps_si128(tv),20);;(tw)=_mm_max_ps(tw,_mm_castsi128_ps((ht)));(tw)=/////
_mm_min_ps(tw,_mm_castsi128_ps((hu)));ua=_mm_srli_epi32(_mm_castps_si128(tw),20
);;{hk uf,ug,uh,ui;uf.ab=tx;ug.ab=ty;uh.ab=tz;ui.ab=ua;uf.l[0]=(gu-(127-13)*8)[
uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(gu-(127-13)*8)[uf.u[2]];uf.l
[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8)[ug.u[0]];ug.l[1]=(gu-(127-
13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug.l[3]=(gu-(127-13)*8)[ug.u[3
]];uh.l[0]=(gu-(127-13)*8)[uh.u[0]];uh.l[1]=(gu-(127-13)*8)[uh.u[1]];uh.l[2]=(
gu-(127-13)*8)[uh.u[2]];uh.l[3]=(gu-(127-13)*8)[uh.u[3]];ui.l[0]=(gu-(127-13)*8
)[ui.u[0]];ui.l[1]=(gu-(127-13)*8)[ui.u[1]];ui.l[2]=(gu-(127-13)*8)[ui.u[2]];ui
.l[3]=(gu-(127-13)*8)[ui.u[3]];tx=uf.ab;ty=ug.ab;tz=uh.ab;ua=ui.ab;};{__m128i//
uj;uj=_mm_srli_epi32(_mm_castps_si128(tt),12);(uj)=_mm_and_si128(uj,(hv));(uj)=
_mm_or_si128(uj,(hw));(tx)=_mm_madd_epi16(tx,uj);tx=_mm_srli_epi32(tx,16);};{//
__m128i uk;uk=_mm_srli_epi32(_mm_castps_si128(tu),12);(uk)=_mm_and_si128(uk,(hv
));(uk)=_mm_or_si128(uk,(hw));(ty)=_mm_madd_epi16(ty,uk);ty=_mm_srli_epi32(ty,
16);};{__m128i ul;ul=_mm_srli_epi32(_mm_castps_si128(tv),12);(ul)=_mm_and_si128
(ul,(hv));(ul)=_mm_or_si128(ul,(hw));(tz)=_mm_madd_epi16(tz,ul);tz=////////////
_mm_srli_epi32(tz,16);};{__m128i um;um=_mm_srli_epi32(_mm_castps_si128(tw),12);
(um)=_mm_and_si128(um,(hv));(um)=_mm_or_si128(um,(hw));(ua)=_mm_madd_epi16(ua,
um);ua=_mm_srli_epi32(ua,16);};ty=_mm_packs_epi32(ty,tx);ua=_mm_packs_epi32(ua,
tz);tx=_mm_unpacklo_epi16(ty,ua);tz=_mm_unpackhi_epi16(ty,ua);ty=//////////////
_mm_unpacklo_epi16(tx,tz);ua=_mm_unpackhi_epi16(tx,tz);ty=_mm_packus_epi16(ty,
ua);_mm_storeu_si128((__m128i*)(tq),ty);;tp+=16;tq+=16;if(tq<=tr)continue;if(tq
==(tr+16))break;tq=tr;tp=ts;}return;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tp));tq[0-4]=gv(tp[1]);tq[1-4]=gv(
tp[0]);tq[2-4]=gv(tp[3]);tq[3-4]=gv(tp[2]);tq+=4;tp+=4;}tq-=4;_Pragma(/////////
"GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(""::"r"(tp));tq[0]=
gv(tp[1]);tq[1]=gv(tp[0]);tq+=2;tp+=2;}}static float*mc(float*tn,int to,void///
const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+to;unsigned char const*
ts=(unsigned char const*)tp;tq+=4;while(tq<=tr){tq[0-4]=gs[ts[1]];tq[1-4]=((///
float)ts[0])*3.9215689e-03f;tq[2-4]=gs[ts[1+2]];tq[3-4]=((float)ts[0+2])*//////
3.9215689e-03f;ts+=4;tq+=4;}tq-=4;if(tq<tr){tq[0]=gs[ts[1]];tq[1]=((float)ts[0]
)*3.9215689e-03f;}return tr;}static void md(void*tn,int to,float const*tp){////
unsigned char*__restrict__ tq=(unsigned char*)tn;unsigned char*tr=((unsigned///
char*)tq)+to;if(to>=16){float const*ts=tp+to-16;tr-=16;for(;;){__m128 tt,tu,tv,
tw;__m128i tx,ty,tz,ua;__asm__(""::"r"(tp));(tt)=_mm_loadu_ps((float const*)((
tp)));(tu)=_mm_loadu_ps((float const*)((tp)+4));(tv)=_mm_loadu_ps((float const*
)((tp)+8));(tw)=_mm_loadu_ps((float const*)((tp)+12));{__m128 ub,uc,ud,ue;ub=//
_mm_unpacklo_ps(tt,tu);ud=_mm_unpacklo_ps(tv,tw);uc=_mm_unpackhi_ps(tt,tu);ue=
_mm_unpackhi_ps(tv,tw);tt=_mm_movelh_ps(ub,ud);tu=_mm_movehl_ps(ud,ub);tv=/////
_mm_movelh_ps(uc,ue);tw=_mm_movehl_ps(ue,uc);};(tt)=_mm_max_ps(tt,/////////////
_mm_castsi128_ps((ht)));(tt)=_mm_min_ps(tt,_mm_castsi128_ps((hu)));tx=/////////
_mm_srli_epi32(_mm_castps_si128(tt),20);;(tu)=_mm_add_ps((hq),_mm_mul_ps((hm),
tu));(tu)=_mm_max_ps(tu,_mm_setzero_ps());(tu)=_mm_min_ps(tu,(hm));(ty)=///////
_mm_cvttps_epi32(tu);;(tv)=_mm_max_ps(tv,_mm_castsi128_ps((ht)));(tv)=/////////
_mm_min_ps(tv,_mm_castsi128_ps((hu)));tz=_mm_srli_epi32(_mm_castps_si128(tv),20
);;(tw)=_mm_add_ps((hq),_mm_mul_ps((hm),tw));(tw)=_mm_max_ps(tw,_mm_setzero_ps(
));(tw)=_mm_min_ps(tw,(hm));(ua)=_mm_cvttps_epi32(tw);;{hk uf,ug;uf.ab=tx;ug.ab
=tz;uf.l[0]=(gu-(127-13)*8)[uf.u[0]];uf.l[1]=(gu-(127-13)*8)[uf.u[1]];uf.l[2]=(
gu-(127-13)*8)[uf.u[2]];uf.l[3]=(gu-(127-13)*8)[uf.u[3]];ug.l[0]=(gu-(127-13)*8
)[ug.u[0]];ug.l[1]=(gu-(127-13)*8)[ug.u[1]];ug.l[2]=(gu-(127-13)*8)[ug.u[2]];ug
.l[3]=(gu-(127-13)*8)[ug.u[3]];tx=uf.ab;tz=ug.ab;};{__m128i uh;uh=/////////////
_mm_srli_epi32(_mm_castps_si128(tt),12);(uh)=_mm_and_si128(uh,(hv));(uh)=//////
_mm_or_si128(uh,(hw));(tx)=_mm_madd_epi16(tx,uh);tx=_mm_srli_epi32(tx,16);};{//
__m128i ui;ui=_mm_srli_epi32(_mm_castps_si128(tv),12);(ui)=_mm_and_si128(ui,(hv
));(ui)=_mm_or_si128(ui,(hw));(tz)=_mm_madd_epi16(tz,ui);tz=_mm_srli_epi32(tz,
16);};ty=_mm_packs_epi32(ty,tx);ua=_mm_packs_epi32(ua,tz);tx=_mm_unpacklo_epi16
(ty,ua);tz=_mm_unpackhi_epi16(ty,ua);ty=_mm_unpacklo_epi16(tx,tz);ua=//////////
_mm_unpackhi_epi16(tx,tz);ty=_mm_packus_epi16(ty,ua);_mm_storeu_si128((__m128i*
)(tq),ty);;tq+=16;tp+=16;if(tq<=tr)continue;if(tq==(tr+16))break;tq=tr;tp=ts;}
return;}_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float uj;__asm__(""::
"r"(tp));tq[1]=gv(tp[0]);uj=tp[1]*255.0f+0.5f;for(;;){if((uj)<(0))(uj)=(0);if((
uj)>(255))(uj)=(255);break;};tq[0]=(unsigned char)uj;tq+=2;tp+=2;}while(tq<tr);
}static float*me(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*
tr=(float*)tq+to;unsigned short const*ts=(unsigned short const*)tp;unsigned////
short const*tt=ts+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128 tx,ty;//
__asm__(""::"r"(tq));(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i tz=///
_mm_setzero_si128();tv=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz);}
;(tx)=_mm_cvtepi32_ps(tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_mul_ps(tx,(hp));(ty
)=_mm_mul_ps(ty,(hp));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tx),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ty),(1<<0)+(0<<2)+(3<<4)+(2<<6)));_mm_storeu_ps((float*)(tq+0)
,tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;ts+=8;if(tq<=tr)continue;if(tq==(tr
+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=((float)(ts[1]))*////
1.5259022e-05f;tq[1-4]=((float)(ts[0]))*1.5259022e-05f;tq[2-4]=((float)(ts[3]))
*1.5259022e-05f;tq[3-4]=((float)(ts[2]))*1.5259022e-05f;tq+=4;ts+=4;}tq-=4;////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){__asm__(""::"r"(tq))
;tq[0]=((float)(ts[1]))*1.5259022e-05f;tq[1]=((float)(ts[0]))*1.5259022e-05f;tq
+=2;ts+=2;}return tr;}static void mf(void*tn,int to,float const*tp){unsigned///
short*__restrict__ tq=(unsigned short*)tn;unsigned short*tr=((unsigned short*)
tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){__m128 tt,tu;/////
__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_mul_ps((hn),//////////
_mm_loadu_ps((float const*)(tp))));(tu)=_mm_add_ps((hq),_mm_mul_ps((hn),///////
_mm_loadu_ps((float const*)(tp+4))));(tt)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(tt),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(tu)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128i//
tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,(hn)),_mm_setzero_ps()));tx=
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),_mm_setzero_ps()));tw=/////////
_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=_mm_packs_epi32(tw,tx);tv=/////
_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq),tv);tp+=4*2;tq+=4*2;if(
tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}return;}}tq+=4;_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__m128 ty;__m128i tz;//////
__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp));(ty)=_mm_add_ps((hq)
,_mm_mul_ps((hn),ty));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
ty),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128i ua,ub;ua=_mm_cvttps_epi32(_mm_max_ps
(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=_mm_cvttps_epi32(_mm_max_ps(////////
_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32
(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=_mm_sub_epi16(tz,gz);};_mm_storel_epi64((
__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(///////
"GCC novector")while(tq<tr){__m128 uc;__asm__(""::"r"(tp));(uc)=_mm_add_ss((hq)
,_mm_mul_ss((hn),_mm_load_ss((float const*)(tp+1))));tq[0]=((unsigned short)///
_mm_cvtsi128_si32(_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(uc,(hn)),/////////////
_mm_setzero_ps()))));(uc)=_mm_add_ss((hq),_mm_mul_ss((hn),_mm_load_ss((float///
const*)(tp+0))));tq[1]=((unsigned short)_mm_cvtsi128_si32(_mm_cvttps_epi32(////
_mm_max_ps(_mm_min_ps(uc,(hn)),_mm_setzero_ps()))));tq+=2;tp+=2;}}static float*
mg(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*tr=(float*)tq+
to;unsigned short const*ts=(unsigned short const*)tp;unsigned short const*tt=ts
+to-8;if(to>=8){tr-=8;for(;;){__m128i tu,tv,tw;__m128 tx,ty;__asm__(""::"r"(tq)
);(tu)=_mm_loadu_si128((__m128i const*)(ts));{__m128i tz=_mm_setzero_si128();tv
=_mm_unpacklo_epi16(tu,tz);tw=_mm_unpackhi_epi16(tu,tz);};(tx)=_mm_cvtepi32_ps(
tv);(ty)=_mm_cvtepi32_ps(tw);(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(//////////
_mm_castps_si128(tx),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(ty)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ty),(1<<0)+(0<<2)+(3<<4)+(2<<6)));//////////
_mm_storeu_ps((float*)(tq+0),tx);_mm_storeu_ps((float*)(tq+4),ty);tq+=8;ts+=8;
if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4
]=((float)(ts[1]));tq[1-4]=((float)(ts[0]));tq[2-4]=((float)(ts[3]));tq[3-4]=((
float)(ts[2]));tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector"
)while(tq<tr){__asm__(""::"r"(tq));tq[0]=((float)(ts[1]));tq[1]=((float)(ts[0])
);tq+=2;ts+=2;}return tr;}static void mh(void*tn,int to,float const*tp){///////
unsigned short*__restrict__ tq=(unsigned short*)tn;unsigned short*tr=((unsigned
short*)tq)+to;{if(to>=4*2){float const*ts=tp+to-4*2;tr-=4*2;for(;;){__m128 tt,
tu;__m128i tv;__asm__(""::"r"(tp));(tt)=_mm_add_ps((hq),_mm_loadu_ps((float////
const*)(tp)));(tu)=_mm_add_ps((hq),_mm_loadu_ps((float const*)(tp+4)));(tt)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<
2)+(3<<4)+(2<<6)));{__m128i tw,tx;tw=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tt,
(hn)),_mm_setzero_ps()));tx=_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(tu,(hn)),///
_mm_setzero_ps()));tw=_mm_sub_epi32(tw,gy);tx=_mm_sub_epi32(tx,gy);tv=/////////
_mm_packs_epi32(tw,tx);tv=_mm_sub_epi16(tv,gz);};_mm_storeu_si128((__m128i*)(tq
),tv);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+4*2))break;tq=tr;tp=ts;}///
return;}}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){////
__m128 ty;__m128i tz;__asm__(""::"r"(tp));(ty)=_mm_loadu_ps((float const*)(tp))
;(ty)=_mm_add_ps((hq),ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(ty),(1<<0)+(0<<2)+(3<<4)+(2<<6)));{__m128i ua,ub;ua=//////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ub=/////////
_mm_cvttps_epi32(_mm_max_ps(_mm_min_ps(ty,(hn)),_mm_setzero_ps()));ua=/////////
_mm_sub_epi32(ua,gy);ub=_mm_sub_epi32(ub,gy);tz=_mm_packs_epi32(ua,ub);tz=/////
_mm_sub_epi16(tz,gz);};_mm_storel_epi64((__m128i*)(tq-4),(tz));tq+=4;tp+=4;}tq
-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){float uc;__asm__
(""::"r"(tp));uc=tp[1]+0.5f;for(;;){if((uc)<(0))(uc)=(0);if((uc)>(65535))(uc)=(
65535);break;};tq[0]=(unsigned short)uc;uc=tp[0]+0.5f;for(;;){if((uc)<(0))(uc)=
(0);if((uc)>(65535))(uc)=(65535);break;};tq[1]=(unsigned short)uc;tq+=2;tp+=2;}
}static float*mi(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*
tr=(float*)tq+to;he const*ts=(he const*)tp;if(to>=8){he const*tt=ts+to-8;tr-=8;
for(;;){__asm__(""::"r"(tq));hh(tq,ts);{__m128 tu,tv;(tu)=_mm_loadu_ps((float//
const*)(tq));(tv)=_mm_loadu_ps((float const*)(tq+4));(tu)=_mm_castsi128_ps(////
_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(tv)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));_mm_storeu_ps((float*)(tq),tu);_mm_storeu_ps((float*)(tq+4),tv);}tq+=8;
ts+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;ts=tt;}return tr+8;}tq+=4;//
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){__asm__(""::"r"(tq)
);tq[0-4]=hf(ts[1]);tq[1-4]=hf(ts[0]);tq[2-4]=hf(ts[3]);tq[3-4]=hf(ts[2]);tq+=4
;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<tr){//////
__asm__(""::"r"(tq));tq[0]=hf(ts[1]);tq[1]=hf(ts[0]);tq+=2;ts+=2;}return tr;}//
static void mj(void*tn,int to,float const*tp){he*__restrict__ tq=(he*)tn;he*tr=
((he*)tq)+to;if(to>=8){float const*ts=tp+to-8;tr-=8;for(;;){__asm__(""::"r"(tp)
);{__m128 tt[2];(tt[0])=_mm_loadu_ps((float const*)(tp));(tt[1])=_mm_loadu_ps((
float const*)(tp+4));(tt[0])=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(tt[0]),(1<<0)+(0<<2)+(3<<4)+(2<<6)));(tt[1])=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tt[1]),(1<<0)+(0<<2)+(3<<4)+(2<<6)));hj(tq,(
float*)tt);}tp+=8;tq+=8;if(tq<=tr)continue;if(tq==(tr+8))break;tq=tr;tp=ts;}///
return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){/////
__asm__(""::"r"(tq));tq[0-4]=hg(tp[1]);tq[1-4]=hg(tp[0]);tq[2-4]=hg(tp[3]);tq[3
-4]=hg(tp[2]);tq+=4;tp+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")
while(tq<tr){__asm__(""::"r"(tq));tq[0]=hg(tp[1]);tq[1]=hg(tp[0]);tq+=2;tp+=2;}
}static float*mk(float*tn,int to,void const*tp){float*__restrict__ tq=tn;float*
tr=(float*)tq+to;float const*ts=(float const*)tp;if(to>=16){float const*tt=ts+
to-16;tr-=16;for(;;){__asm__(""::"r"(tq));{__m128 tu,tv,tw,tx;(tu)=_mm_loadu_ps
((float const*)(ts));(tv)=_mm_loadu_ps((float const*)(ts+4));(tw)=_mm_loadu_ps(
(float const*)(ts+8));(tx)=_mm_loadu_ps((float const*)(ts+12));(tu)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(1<<0)+(0<<
2)+(3<<4)+(2<<6)));(tw)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tw)
,(1<<0)+(0<<2)+(3<<4)+(2<<6)));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(////////
_mm_castps_si128(tx),(1<<0)+(0<<2)+(3<<4)+(2<<6)));_mm_storeu_ps((float*)(tq),
tu);_mm_storeu_ps((float*)(tq+4),tv);_mm_storeu_ps((float*)(tq+8),tw);/////////
_mm_storeu_ps((float*)(tq+12),tx);}tq+=16;ts+=16;if(tq<=tr)continue;if(tq==(tr+
16))break;tq=tr;ts=tt;}return tr+16;}tq+=4;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")while(tq<=tr){__asm__(""::"r"(tq));tq[0-4]=ts[1];tq[1-4]=ts[0];
tq[2-4]=ts[3];tq[3-4]=ts[2];tq+=4;ts+=4;}tq-=4;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")while(tq<tr){__asm__(""::"r"(tq));tq[0]=ts[1];tq[1]=ts[0];tq+=2;
ts+=2;}return tr;}static void ml(void*tn,int to,float const*tp){float*/////////
__restrict__ tq=(float*)tn;float*tr=((float*)tq)+to;if(to>=(4*2)){float const*
ts=tp+to-(4*2);tr-=(4*2);for(;;){__m128 tt,tu;__asm__(""::"r"(tp));(tt)=///////
_mm_loadu_ps((float const*)(tp));(tu)=_mm_loadu_ps((float const*)(tp+4));(tt)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));(tu)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(0<<
2)+(3<<4)+(2<<6)));_mm_storeu_ps((float*)(tq),tt);_mm_storeu_ps((float*)(tq+4),
tu);tp+=4*2;tq+=4*2;if(tq<=tr)continue;if(tq==(tr+(4*2)))break;tq=tr;tp=ts;}///
return;}tq+=4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tq<=tr){/////
__m128 tv;__asm__(""::"r"(tp));(tv)=_mm_loadu_ps((float const*)(tp));(tv)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(1<<0)+(0<<2)+(3<<4)+(2
<<6)));_mm_storeu_ps((float*)(tq-4),tv);tq+=4;tp+=4;}tq-=4;_Pragma(////////////
"GCC unroll 1")_Pragma("GCC novector")while(tq<tr){float tw;__asm__(""::"r"(tp)
);tw=tp[1];;;tq[0]=tw;tw=tp[0];;;tq[1]=tw;tq+=2;tp+=2;}}static void mm(float*tn
,int to){float*__restrict__ tp=tn;float const*tq=tn+(to/4)*7;float*__restrict__
tr=(float*)tq-to;tr+=8;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tr<=
tq){__m128 ts,tt,tu,tv,tw,tx;__asm__(""::"r"(tr));(ts)=_mm_loadu_ps((float/////
const*)(tr-8));(tu)=_mm_loadu_ps((float const*)(tr-8+4));(tt)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ts),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tv)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tw)=_mm_mul_ps(tt,ts);(tx)=_mm_mul_ps(tv,tu);_mm_storeu_ps((float*)(tp)
,ts);_mm_storeu_ps((float*)(tp+4),tw);_mm_storeu_ps((float*)(tp+7),tu);////////
_mm_storeu_ps((float*)(tp+7+4),tx);tr+=8;tp+=14;}tr-=8;if(tr<tq){__m128 ty,tz,
ua;__asm__(""::"r"(tr));(ty)=_mm_loadu_ps((float const*)(tr));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_mul_ps(tz,ty);_mm_storeu_ps((float*)(tp),ty);_mm_storeu_ps((///
float*)(tp+4),ua);tr+=4;tp+=7;}}static void mn(float*tn,int to){float*/////////
__restrict__ tp=tn;float const*tq=tn+(to/2)*3;float*__restrict__ tr=(float*)tq-
to;tr+=8;if(tr<=tq){_Pragma("GCC unroll 1")_Pragma("GCC novector")do{__m128 ts,
tt,tu,tv,tw,tx;__asm__(""::"r"(tr));(ts)=_mm_loadu_ps((float const*)(tr-8));(tu
)=_mm_loadu_ps((float const*)(tr-8+4));(tw)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ts),(1<<0)+(1<<2)+(3<<4)+(3<<6)));(tx)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tu),(1<<0)+(1<<2)+(3<<4)+(3<<6)));(tt)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ts),(0<<0)+(0<<2)+(2<<4)+(2
<<6)));(tv)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(0<<0)+(0<<
2)+(2<<4)+(2<<6)));(tw)=_mm_mul_ps(tw,tt);(tx)=_mm_mul_ps(tx,tv);//////////////
_mm_storel_epi64((__m128i*)(tp),_mm_castps_si128(ts));_mm_storeu_ps((float*)(tp
+2),tw);_mm_storeh_pd((double*)(tp+3),_mm_castps_pd(ts));_mm_storel_epi64((////
__m128i*)(tp+6),_mm_castps_si128(tu));_mm_storeu_ps((float*)(tp+8),tx);////////
_mm_storeh_pd((double*)(tp+9),_mm_castps_pd(tu));tr+=8;tp+=12;}while(tr<=tq);}
tr-=8;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tr<tq){float ty=tr[0]
,tz=tr[1];__asm__(""::"r"(tr));tp[0]=ty;tp[1]=tz;tp[2]=ty*tz;tp+=3;tr+=2;}}////
static void mo(float*tn,int to){float*__restrict__ tp=tn;float*__restrict__ tq=
tn;float const*tr=tn+to;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
ts=tq[3];__m128 tt,tu;__asm__(""::"r"(tp));if(ts<((float)1/(1<<20)/(1<<20)/(1<<
20)/(1<<20)/(1<<20)/(1<<20))){(tt)=_mm_loadu_ps((float const*)(tq));///////////
_mm_storeu_ps((float*)(tp),tt);}else{(tu)=_mm_set_ps1(1.0f/ts);(tt)=///////////
_mm_loadu_ps((float const*)(tq+4));(tt)=_mm_mul_ps(tt,tu);_mm_storeu_ps((float*
)(tp),tt);tp[3]=ts;}tq+=7;tp+=4;}while(tp<tr);}static void mp(float*tn,int to){
float*__restrict__ tp=tn;float*__restrict__ tq=tn;float const*tr=tn+to;do{float
ts=tq[1];tp[0]=tq[0];if(ts>=((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(
1<<20)))tp[0]=tq[2]/ts;tp[1]=ts;tq+=3;tp+=2;}while(tp<tr);}static void mq(float
*tn,int to){float*__restrict__ tp=tn;float const*tq=tn+to;{tp+=2*4;_Pragma(////
"GCC unroll 1")_Pragma("GCC novector")while(tp<=tq){__m128 tr,ts,tt,tu;__asm__(
""::"r"(tp));(tr)=_mm_loadu_ps((float const*)(tp-2*4));(tt)=_mm_loadu_ps((float
const*)(tp-2*4+4));(ts)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(///
_mm_movehl_ps((hr),tr)),(1<<0)+(1<<2)+(1<<4)+(2<<6)));(tu)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(_mm_movehl_ps((hr),tt)),(1<<0)+(1<<2)+(1<<4)
+(2<<6)));(tr)=_mm_mul_ps(tr,ts);(tt)=_mm_mul_ps(tt,tu);_mm_storeu_ps((float*)(
tp-2*4),tr);_mm_storeu_ps((float*)(tp-2*4+4),tt);tp+=2*4;}tp-=2*4;if(tp<tq){///
__m128 tv,tw;(tv)=_mm_loadu_ps((float const*)(tp));(tw)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(_mm_movehl_ps((hr),tv)),(1<<0)+(1<<2)+(1<<4)
+(2<<6)));(tv)=_mm_mul_ps(tv,tw);_mm_storeu_ps((float*)(tp),tv);tp+=4;}}}static
void mr(float*tn,int to){float*__restrict__ tp=tn;float const*tq=tn+to;tp+=2*4;
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<=tq){__m128 tr,ts,tt,tu;
__asm__(""::"r"(tp));(tr)=_mm_loadu_ps((float const*)(tp-2*4));(tt)=///////////
_mm_loadu_ps((float const*)(tp-2*4+4));(ts)=_mm_or_ps(_mm_castsi128_ps(////////
_mm_srli_epi64(_mm_castps_si128(tr),32)),gx);(tu)=_mm_or_ps(_mm_castsi128_ps(//
_mm_srli_epi64(_mm_castps_si128(tt),32)),gx);(tr)=_mm_mul_ps(tr,ts);(tt)=//////
_mm_mul_ps(tt,tu);_mm_storeu_ps((float*)(tp-2*4),tr);_mm_storeu_ps((float*)(tp-
2*4+4),tt);tp+=2*4;}tp-=2*4;_Pragma("GCC unroll 1")_Pragma("GCC novector")while
(tp<tq){float tv=tp[1];__asm__(""::"r"(tp));tp[0]*=tv;tp+=2;}}static void ms(//
float*tn,int to){float*__restrict__ tp=tn;float const*tq=tn+to;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")do{float tr=tp[3];__m128 ts,tt;__asm__(""
::"r"(tp));if(tr>=((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20))){(
tt)=_mm_set_ps1(1.0f/tr);(ts)=_mm_loadu_ps((float const*)(tp));(ts)=_mm_mul_ps(
ts,tt);_mm_storeu_ps((float*)(tp),ts);tp[3]=tr;}tp+=4;}while(tp<tq);}static////
void mt(float*tn,int to){float*__restrict__ tp=tn;float const*tq=tn+to;do{float
tr=tp[1];if(tr>=((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))tp[0
]/=tr;tp+=2;}while(tp<tq);}static void mu(float*tn,int to){float*__restrict__//
tp=tn;float const*tq=tn+to;tq-=24;_Pragma("GCC unroll 1")_Pragma("GCC novector"
)while(tp<=tq){__m128 tr,ts,tt,tu,tv,tw,tx;float ty,tz;__asm__(""::"r"(tp));(tr
)=_mm_loadu_ps((float const*)(tp));(ts)=_mm_loadu_ps((float const*)(tp+3));(tt)
=_mm_loadu_ps((float const*)(tp+6));(tu)=_mm_loadu_ps((float const*)(tp+9));(tv
)=_mm_loadu_ps((float const*)(tp+12));(tw)=_mm_loadu_ps((float const*)(tp+15));
(tx)=_mm_loadu_ps((float const*)(tp+18));tr=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(tr),(2<<0)+(1<<2)+(0<<4)+(3<<6)));ts=_mm_castsi128_ps(////////
_mm_shuffle_epi32(_mm_castps_si128(ts),(2<<0)+(1<<2)+(0<<4)+(3<<6)));tt=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tt),(2<<0)+(1<<2)+(0<<4)+(3
<<6)));tu=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tu),(2<<0)+(1<<2)
+(0<<4)+(3<<6)));tv=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tv),(2
<<0)+(1<<2)+(0<<4)+(3<<6)));tw=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(tw),(2<<0)+(1<<2)+(0<<4)+(3<<6)));tx=_mm_castsi128_ps(////////
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(1<<2)+(0<<4)+(3<<6)));//////////
_mm_storeu_ps((float*)(tp),tr);ty=tp[21];_mm_storeu_ps((float*)(tp+3),ts);tz=tp
[23];_mm_storeu_ps((float*)(tp+6),tt);_mm_storeu_ps((float*)(tp+9),tu);////////
_mm_storeu_ps((float*)(tp+12),tv);_mm_storeu_ps((float*)(tp+15),tw);///////////
_mm_storeu_ps((float*)(tp+18),tx);tp[21]=tz;tp[23]=ty;tp+=24;}tq+=24;_Pragma(//
"GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float ua=tp[0];__asm__(""::
"r"(tp));tp[0]=tp[2];tp[2]=ua;tp+=3;}}static void mv(fz const*tn,int to,float*
tp){int tq=tn->bd;int tr=tn->be;int ts=gd[tn->ai]*tq;fs tt=tn->l.aj;fs tu=tn->u
.aj;int tv=iw(tu,to,tn->u.ad.l);const void*tw=((char*)tn->ab)+(size_t)tv*(/////
size_t)tn->ad;gg const*tx=tn->an.ab;float*ty=tp-tn->an.l.l*tr;float*tz=0;assert
(!(tu==STBIR_EDGE_ZERO&&(to<0||to>=tn->u.ad.l)));do{float*ua;void const*ub;////
float*uc;int ud;int ue;if(tx->u<tx->l)break;ue=tx->u+1-tx->l;ua=ty+tx->l*tr;uc=
ty+(tx->u+1)*tr;ud=ue*tq;ub=((char*)tw)+tx->ab*ts;if(tn->ak){ub=tn->ak(((char*)
uc)-(ue*ts)+((tn->ai!=STBIR_TYPE_FLOAT)?(sizeof(float)*3):0),tw,ue,tx->ab,tv,tn
->al);};tz=tn->aq((float*)uc-ud,ud,ub);;if(tn->ar){;tn->ar(ua,ud);;}++tx;}while
(tx<=(&tn->an.ab[1]));if((tt==STBIR_EDGE_WRAP)&&(tn->an.u[0]|tn->an.u[1])){int
uf,ug[2];int uh=tn->l.ad.l;ug[0]=-tn->an.u[0];ug[1]=uh;for(uf=0;uf<2;uf++){int
ui=tn->an.u[uf];if(ui){int uj=ug[uf];float*uk=ty+uj*tr;float const*ul=ty+iw(tt,
uj,uh)*tr;hx(uk,ul,ui*tr*sizeof(float));if(uf==1)tz=uk+ui*tr;}}}tz[0]=0.0f;tz[1
]=0.0f;}static void mw(float*tn,unsigned int to,float const*tp,ge const*tq,////
float const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;float const*
tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_load_ss((float const*)(tw));(
tx)=_mm_mul_ss(ty,_mm_load_ss((float const*)(tv)));;_mm_store_ss((float*)(tu),
tx);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void mx(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*1;float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(ty)=////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw)));(tz)=_mm_castsi128_ps(///////
_mm_loadl_epi64((__m128i*)(tv)));(tx)=_mm_mul_ps(ty,tz);(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ss(tx,ty);;_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<
tt);}static void my(float*tn,unsigned int to,float const*tp,ge const*tq,float//
const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;float const*
tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw)
);(tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));(ty)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));(tx)=_mm_add_ss(tx,ty);(tx)=_mm_add_ss(tx,tz);;_mm_store_ss((float*)(tu)
,tx);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void mz(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*1;float const*tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=///////
_mm_loadu_ps((float const*)(tw));(tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)
(tv)));;(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3
<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(/////////////
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ss(tx,ty);_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<tt
);}static void na(float*tn,unsigned int to,float const*tp,ge const*tq,float////
const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;float const*
tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw));(
tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));;{__m128 tz;(ty)=//////////
_mm_load_ss((float const*)(tw+(4)));(tz)=_mm_load_ss((float const*)(tv+(4)));(
tx)=_mm_add_ps(tx,_mm_mul_ps(tz,ty));};(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(tx)=_mm_add_ss(tx,ty);_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;
}while(tu<tt);}static void nb(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;////
float const*tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float///
const*)(tw));(tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));;{__m128 tz;(
ty)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(4))));(tz)=_mm_castsi128_ps
(_mm_loadl_epi64((__m128i*)(tv+(4))));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,ty));};(
ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4
)+(1<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ss(tx,ty);/////
_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void nc(
float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){////
float const*tt=tn+to*1;float*__restrict__ tu=tn;__m128 tv;(tv)=_mm_loadu_ps((//
float const*)(hl+3));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float///
const*tw=tp+tq->l*1;float const*tx=tr;__m128 ty,tz;__asm__(""::"r"(tw));(tz)=//
_mm_loadu_ps((float const*)(tx));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)
(tw)));;(tz)=_mm_loadu_ps((float const*)(tx+(4)));(tz)=_mm_and_ps(tz,tv);(ty)=
_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tw+(4)))));;(tz)=//////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));(ty)=_mm_add_ps(ty,tz);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////
_mm_castps_si128(ty),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=_mm_add_ss(ty,tz);/////
_mm_store_ss((float*)(tu),ty);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void nd(
float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){////
float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*1;float const*tw=tr;__m128 tx,ty;////
__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw));(tx)=_mm_mul_ps(ty,
_mm_loadu_ps((float const*)(tv)));;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((////
float const*)(tw+(4)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ty,_mm_loadu_ps((float////
const*)(tv+(4)))));;(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx
),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ss(tx,ty);_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<tt
);}static void ne(float*tn,unsigned int to,float const*tp,ge const*tq,float////
const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;float const*
tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw));(
tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));;__asm__(""::"r"(tv));(ty)=
_mm_loadu_ps((float const*)(tw+(4)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv+(4)))));;{__m128 tz;(ty)=_mm_load_ss((float/////
const*)(tw+(8)));(tz)=_mm_load_ss((float const*)(tv+(8)));(tx)=_mm_add_ps(tx,//
_mm_mul_ps(tz,ty));};(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ss(tx,ty);_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<tt
);}static void nf(float*tn,unsigned int to,float const*tp,ge const*tq,float////
const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;float const*
tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw));(
tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));;__asm__(""::"r"(tv));(ty)=
_mm_loadu_ps((float const*)(tw+(4)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv+(4)))));;{__m128 tz;(ty)=_mm_castsi128_ps(//////
_mm_loadl_epi64((__m128i*)(tw+(8))));(tz)=_mm_castsi128_ps(_mm_loadl_epi64((///
__m128i*)(tv+(8))));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,ty));};(ty)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////
_mm_castps_si128(tx),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ss(tx,ty);/////
_mm_store_ss((float*)(tu),tx);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void ng(
float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){////
float const*tt=tn+to*1;float*__restrict__ tu=tn;__m128 tv;(tv)=_mm_loadu_ps((//
float const*)(hl+3));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float///
const*tw=tp+tq->l*1;float const*tx=tr;__m128 ty,tz;__asm__(""::"r"(tw));(tz)=//
_mm_loadu_ps((float const*)(tx));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)
(tw)));;__asm__(""::"r"(tw));(tz)=_mm_loadu_ps((float const*)(tx+(4)));(ty)=///
_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tw+(4)))));;(tz)=//////
_mm_loadu_ps((float const*)(tx+(8)));(tz)=_mm_and_ps(tz,tv);(ty)=_mm_add_ps(ty,
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tw+(8)))));;(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,tz);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),
(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=_mm_add_ss(ty,tz);_mm_store_ss((float*)(tu),
ty);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void nh(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*1;float const*tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=///////
_mm_loadu_ps((float const*)(tw));(tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)
(tv)));;__asm__(""::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw+(4)));(tx)=///
_mm_add_ps(tx,_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv+(4)))));;__asm__(""
::"r"(tv));(ty)=_mm_loadu_ps((float const*)(tw+(8)));(tx)=_mm_add_ps(tx,///////
_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv+(8)))));;(ty)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),
(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ss(tx,ty);_mm_store_ss((float*)(tu),
tx);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void ni(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*1;int tw=((tq->u-tq->l+1)-4+3)>>2;float const*tx=tr;__m128 ty,tz;
__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float const*)(tx));(ty)=_mm_mul_ps(tz,
_mm_loadu_ps((float const*)(tv)));;_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")do{tx+=4;tv+=1*4;__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float//
const*)(tx+(0)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+(0)))));;--tw;}while(tw>0);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////
_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,tz);(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(ty)=_mm_add_ss(ty,tz);_mm_store_ss((float*)(tu),ty);tr+=ts;++tq;tu+=1;;
}while(tu<tt);}static void nj(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*1;float*__restrict__ tu=tn;/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*1;int
tw=((tq->u-tq->l+1)-5+3)>>2;float const*tx=tr;__m128 ty,tz;__asm__(""::"r"(tv))
;(tz)=_mm_loadu_ps((float const*)(tx));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float//
const*)(tv)));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=1*4;
__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float const*)(tx+(0)));(ty)=_mm_add_ps(
ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(0)))));;--tw;}while(tw>0);{///
__m128 ua;(tz)=_mm_load_ss((float const*)(tx+(4)));(ua)=_mm_load_ss((float/////
const*)(tv+(4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,tz));};(tz)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,tz);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),
(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=_mm_add_ss(ty,tz);_mm_store_ss((float*)(tu),
ty);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void nl(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*1;int tw=((tq->u-tq->l+1)-6+3)>>2;float const*tx=tr;__m128 ty,tz;
__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float const*)(tx));(ty)=_mm_mul_ps(tz,
_mm_loadu_ps((float const*)(tv)));;_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")do{tx+=4;tv+=1*4;__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float//
const*)(tx+(0)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+(0)))));;--tw;}while(tw>0);{__m128 ua;(tz)=_mm_castsi128_ps(_mm_loadl_epi64(
(__m128i*)(tx+(4))));(ua)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4))))
;(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,tz));};(tz)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,tz);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),
(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=_mm_add_ss(ty,tz);_mm_store_ss((float*)(tu),
ty);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static void nm(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*1;float*
__restrict__ tu=tn;__m128 tv;(tv)=_mm_loadu_ps((float const*)(hl+3));;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tw=tp+tq->l*1;int tx=((tq
->u-tq->l+1)-7+3)>>2;float const*ty=tr;__m128 tz,ua;__asm__(""::"r"(tw));(ua)=
_mm_loadu_ps((float const*)(ty));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)
(tw)));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{ty+=4;tw+=1*4;__asm__(
""::"r"(tw));(ua)=_mm_loadu_ps((float const*)(ty+(0)));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tw+(0)))));;--tx;}while(tx>0);(ua)=//
_mm_loadu_ps((float const*)(ty+(4)));(ua)=_mm_and_ps(ua,tv);(tz)=_mm_add_ps(tz,
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tw+(4)))));;(ua)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tz),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tz)=/////
_mm_add_ps(tz,ua);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),
(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tz)=_mm_add_ss(tz,ua);_mm_store_ss((float*)(tu),
tz);tr+=ts;++tq;tu+=1;;}while(tu<tt);}static gn*nn[4]={ni,nj,nl,nm,};static gn*
no[12]={mw,mx,my,mz,na,nb,nc,nd,ne,nf,ng,nh,};static void np(float*tn,unsigned
int to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2
;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{////
float const*tv=tp+tq->l*2;float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv)
);(ty)=_mm_load_ss((float const*)(tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ty),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tz)=_mm_castsi128_ps(//////
_mm_loadl_epi64((__m128i*)(tv)));(tx)=_mm_mul_ps(tz,ty);;(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,ty);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static void nq(float*tn,unsigned int to,float const*
tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
2;float const*tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=_mm_castsi128_ps(///
_mm_loadl_epi64((__m128i*)(tw)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ty),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv)));;(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void nr(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;float const*
tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(tz)=_mm_loadu_ps((float const*)(
tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(0<<0)+(0<<2)
+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv)));(ty)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ua)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+4)));(tx)=_mm_add_ps
(tx,_mm_mul_ps(ua,ty));;(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,ty);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void ns(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;float const*
tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(
tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)
+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;(tx)=_mm_add_ps(
tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<
2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);_mm_storel_epi64((__m128i*)(tu),/////
_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while(tu<tt);}static void nt(float*tn
,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){float const*
tt=tn+to*2;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")do{float const*tv=tp+tq->l*2;float const*tw=tr;__m128 tx,ty,tz,
ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;{__m128 ub;(ua)=
_mm_load_ss((float const*)(tw+(4)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ub)=_mm_castsi128_ps(//////
_mm_loadl_epi64((__m128i*)(tv+(4)*2)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,tz));};
(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);_mm_storel_epi64((///
__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while(tu<tt);}static///
void nu(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int
ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")////
_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;float const*tw=tr;__m128 tx
,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;(ua)=///////////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(4))));(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2))));;(tx)=////
_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),
(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);_mm_storel_epi64((__m128i*
)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while(tu<tt);}static void nv(//
float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){////
float const*tt=tn+to*2;float*__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma
("GCC novector")do{float const*tv=tp+tq->l*2;float const*tw=tr;__m128 tx,ty,tz,
ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;{__m128 ub;(ua)=
_mm_loadu_ps((float const*)(tw+(4)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2))));(tz)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ub)=/////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4)*2+4)));(ty)=_mm_add_ps(ty,//
_mm_mul_ps(ub,tz));};(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(/////////////
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,tz);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static void nw(float*tn,unsigned int to,float const*
tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
2;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2
)+(3<<4)+(3<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;/////
__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3
<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*2+4))));;(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void nx(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;float const*
tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(
tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)
+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;__asm__(""::"r"(
tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=_mm_castsi128_ps(//////////
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2))));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2+4)
)));;{__m128 ub;(ua)=_mm_load_ss((float const*)(tw+(8)));(tz)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ub)=/////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(8)*2)));(tx)=_mm_add_ps(tx,////
_mm_mul_ps(ub,tz));};(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(/////////////
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,tz);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static void ny(float*tn,unsigned int to,float const*
tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
2;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2
)+(3<<4)+(3<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;/////
__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3
<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*2+4))));;(ua)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(8))));(tz)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*2)))
);;(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void nz(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;;_Pragma
("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;float const
*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)
(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2
)+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;__asm__(""::"r"(
tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=_mm_castsi128_ps(//////////
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2))));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3<<4)+(3
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2+4)
)));;{__m128 ub;(ua)=_mm_loadu_ps((float const*)(tw+(8)));(tz)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*2))));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ub)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(8)*2+4)));(ty)=////
_mm_add_ps(ty,_mm_mul_ps(ub,tz));};(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=////
_mm_add_ps(tx,tz);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static void oa(float*tn,unsigned int to,float const*
tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
2;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(1<<4)+(1<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2
)+(3<<4)+(3<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));;/////
__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*2)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(3
<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*2+4))));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(8)));(tz)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(1<<4)+(
1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*2))
));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(
3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+
(8)*2+4))));;(tx)=_mm_add_ps(tx,ty);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,tz);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void ob(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;int tw=((tq
->u-tq->l+1)-4+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(""::"r"(tv));
(ub)=_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));;_Pragma("GCC unroll 1")_Pragma(///////////
"GCC novector")do{tx+=4;tv+=2*4;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float//
const*)(tx+(0)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(
0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((///
float const*)(tv+(0)*2))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////////////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*2+4))));;--tw;}while(tw>0);(ty
)=_mm_add_ps(ty,tz);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty
),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,ua);_mm_storel_epi64((//////
__m128i*)(tu),_mm_castps_si128(ty));tr+=ts;++tq;tu+=2;;}while(tu<tt);}static///
void oc(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int
ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")////
_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;int tw=((tq->u-tq->l+1)-5+3
)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(""::"r"(tv));(ub)=///////////
_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));;_Pragma("GCC unroll 1")_Pragma(///////////
"GCC novector")do{tx+=4;tv+=2*4;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float//
const*)(tx+(0)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(
0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((///
float const*)(tv+(0)*2))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////////////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*2+4))));;--tw;}while(tw>0);{//
__m128 uc;(ub)=_mm_load_ss((float const*)(tx+(4)));(ua)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(uc)=/////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4)*2)));(ty)=_mm_add_ps(ty,////
_mm_mul_ps(uc,ua));};(ty)=_mm_add_ps(ty,tz);(ua)=_mm_castsi128_ps(/////////////
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,ua);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(ty));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static void od(float*tn,unsigned int to,float const*
tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
2;int tw=((tq->u-tq->l+1)-6+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(
""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(/////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));;_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")do{tx+=4;tv+=2*4;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps(
(float const*)(tx+(0)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*2))));(ua)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*2+4))));;--tw;}
while(tw>0);(ub)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tx+(4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*2)))
);;(ty)=_mm_add_ps(ty,tz);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////////////
_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,ua);/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(ty));tr+=ts;++tq;tu+=2;;}while
(tu<tt);}static void oe(float*tn,unsigned int to,float const*tp,ge const*tq,///
float const*tr,int ts){float const*tt=tn+to*2;float*__restrict__ tu=tn;;_Pragma
("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*2;int tw=((tq
->u-tq->l+1)-7+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(""::"r"(tv));
(ub)=_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));;_Pragma("GCC unroll 1")_Pragma(///////////
"GCC novector")do{tx+=4;tv+=2*4;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float//
const*)(tx+(0)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(
0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((///
float const*)(tv+(0)*2))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////////////
_mm_castps_si128(ub),(2<<0)+(2<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*2+4))));;--tw;}while(tw>0);{//
__m128 uc;(ub)=_mm_loadu_ps((float const*)(tx+(4)));(ua)=_mm_castsi128_ps(/////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(1<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*2))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(uc)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4)*2+4)));(tz)=////
_mm_add_ps(tz,_mm_mul_ps(uc,ua));};(ty)=_mm_add_ps(ty,tz);(ua)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ty)=////
_mm_add_ps(ty,ua);_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(ty));tr+=ts;
++tq;tu+=2;;}while(tu<tt);}static gn*og[4]={ob,oc,od,oe,};static gn*oh[12]={np,
nq,nr,ns,nt,nu,nv,nw,nx,ny,nz,oa,};static void oi(float*tn,unsigned int to,////
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(ty)=////
_mm_load_ss((float const*)(tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(///////
_mm_castps_si128(ty),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tz)=_mm_loadu_ps((float////
const*)(tv));(tx)=_mm_mul_ps(tz,ty);;_mm_storel_epi64((__m128i*)(tu),//////////
_mm_castps_si128(tx));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));_mm_store_ss((float*)(tu+2),tx);tr+=ts;++tq;
tu+=3;;}while(tu<tt);}static void oj(float*tn,unsigned int to,float const*tp,ge
const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*__restrict__ tu=tn
;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*3;///
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(tz)=_mm_castsi128_ps
(_mm_loadl_epi64((__m128i*)(tw)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tz),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ua)=_mm_loadu_ps((float////
const*)(tv));(tx)=_mm_mul_ps(ua,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(tz),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=_mm_loadu_ps((float////
const*)(tv+3));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,ty));;_mm_storel_epi64((__m128i
*)(tu),_mm_castps_si128(tx));(tx)=_mm_castsi128_ps(_mm_shuffle_epi32(//////////
_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));_mm_store_ss((float*)(tu+2),
tx);tr+=ts;++tq;tu+=3;;}while(tu<tt);}static void ol(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=
_mm_loadu_ps((float const*)(tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tz)=_mm_loadu_ps((float////
const*)(tv));(tx)=_mm_mul_ps(tz,ty);(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_loadu_ps((float////
const*)(tv+3));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,ty));(ty)=_mm_castsi128_ps(////
_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tz)=/////
_mm_loadu_ps((float const*)(tv+6));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,ty));;/////
_mm_storel_epi64((__m128i*)(tu),_mm_castps_si128(tx));(tx)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_store_ss((float*)(tu+2),tx);tr+=ts;++tq;tu+=3;;}while(tu<tt);}static void//
om(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){
float const*tt=tn+to*3;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,
ua,ub;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw));(ua)=/////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(ty)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8)));;(ua)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)
+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3
<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(tz),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=
_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu+=3;if(tu<tt){//////////
_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((float*)(
tu+2-3),ty);break;;}while(tu<tt);}static void on(float*tn,unsigned int to,float
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*//////
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv));(ub
)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ua)=_mm_load_ss((///
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));;(ua)=_mm_castsi128_ps(///////////////
_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)+(1<<2)+(2<<4)+(3
<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<
2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz)
,(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(
tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),
tx);continue;}(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<
0)+(3<<2)+(0<<4)+(1<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx
));_mm_store_ss((float*)(tu+2-3),ty);break;;}while(tu<tt);}static void oo(float
*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){float////
const*tt=tn+to*3;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")do{float const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,
ua,ub;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw));(ua)=/////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(ty)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8)));;{__m128 uc;/////
__asm__(""::"r"(tv));(ub)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(4))))
;(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0
<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(
4)*3))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1
<<2)+(2<<4)+(2<<6)));(uc)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4)*3+
4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,uc));};(ua)=_mm_castsi128_ps(////////////
_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)+(1<<2)+(2<<4)+(3
<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<
2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz)
,(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(
tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),
tx);continue;}(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<
0)+(3<<2)+(0<<4)+(1<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx
));_mm_store_ss((float*)(tu+2-3),ty);break;;}while(tu<tt);}static void op(float
*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){float////
const*tt=tn+to*3;float*__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma(/////
"GCC novector")do{float const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,
ua,ub;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw));(ua)=/////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(ty)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8)));;{__m128 uc;/////
__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw+(4)));(ua)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3)))
);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2
<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(
4)*3+4))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(uc)=_mm_load_ss((float const*)(tv+(4)*3+8));(tz)=//////
_mm_add_ps(tz,_mm_mul_ps(ua,uc));};(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<
2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(///
_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3
<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,
ua);tr+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(
1<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((
float*)(tu+2-3),ty);break;;}while(tu<tt);}static void oq(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv)
);(ub)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32
(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));(ua)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+8)
)));;(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty
,tx,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=//////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)
+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu
+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((float*)(
tu+2-3),ty);break;;}while(tu<tt);}static void or(float*tn,unsigned int to,float
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*//////
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv));(ub
)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));(ua)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+8)
)));;__asm__(""::"r"(tv));(ua)=_mm_load_ss((float const*)(tw+(8)));(ua)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(8)*3)))
);;(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty,
tx,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)
+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu
+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((float*)(
tu+2-3),ty);break;;}while(tu<tt);}static void os(float*tn,unsigned int to,float
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*//////
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv));(ub
)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));(ua)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+8)
)));;{__m128 uc;__asm__(""::"r"(tv));(ub)=_mm_castsi128_ps(_mm_loadl_epi64((///
__m128i*)(tw+(8))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub
),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((
float const*)(tv+(8)*3))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////////////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(uc)=_mm_castsi128_ps(//////
_mm_loadl_epi64((__m128i*)(tv+(8)*3+4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,uc));
};(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ty,tx
,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(ub)=/////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tz,ty,(0<<0)
+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,ua);tr+=ts;++tq;tu
+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((float*)(
tu+2-3),ty);break;;}while(tu<tt);}static void ot(float*tn,unsigned int to,float
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;float*//////
__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float/////
const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv)
);(ub)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32
(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));(ua)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+8)
)));;{__m128 uc;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw+(8)));
(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<
4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(8)
*3))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<
2)+(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)
(tv+(8)*3+4))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2
<<0)+(2<<2)+(2<<4)+(2<<6)));(uc)=_mm_load_ss((float const*)(tv+(8)*3+8));(tz)=
_mm_add_ps(tz,_mm_mul_ps(ua,uc));};(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(////
_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<
2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(///
_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3
<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,
ua);tr+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(
1<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((
float*)(tu+2-3),ty);break;;}while(tu<tt);}static void ou(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*3;float const*tw=tr;__m128 tx,ty,tz,ua,ub;__asm__(""::"r"(tv)
);(ub)=_mm_loadu_ps((float const*)(tw));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32
(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_mul_ps(ua,////////
_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_mul_ps(ua,/////////
_mm_loadu_ps((float const*)(tv+8)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tw+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ub),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,///////////
_mm_loadu_ps((float const*)(tv+(4)*3))));(ua)=_mm_castsi128_ps(////////////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*3+8)
)));;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tw+(8)));(ua)=//////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(8)*3)))
);(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(2
<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(
8)*3+4))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+
(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float////
const*)(tv+(8)*3+8))));;(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(_mm_shuffle_ps(ty,tx,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<
2)+(1<<4)+(2<<6)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(///
_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(1<<0)+(2<<2)+(3
<<4)+(0<<6)));(tx)=_mm_add_ps(tx,ub);(ua)=_mm_add_ps(ua,tz);(tx)=_mm_add_ps(tx,
ua);tr+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),tx);continue;}(ty)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tx),(2<<0)+(3<<2)+(0<<4)+(
1<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(tx));_mm_store_ss((
float*)(tu+2-3),ty);break;;}while(tu<tt);}static void ov(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*3;int tw=((tq->u-tq->l+1)-4+3)>>2;float const*tx=tr;__m128 ty
,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tx));(ub)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ub)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+4)));(ub)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+8)));;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=3*4;__asm__(""::"r"(tv));(uc
)=_mm_loadu_ps((float const*)(tx+(0)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3))));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+4))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+8)
)));;--tw;}while(tw>0);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)
));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ua,
tz,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ua)=///////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(ty)=_mm_add_ps(ty,uc);(ub)=_mm_add_ps(ub,ua);(ty)=_mm_add_ps(ty,ub);tr
+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),ty);continue;}(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(ty));_mm_store_ss((//
float*)(tu+2-3),tz);break;;}while(tu<tt);}static void ow(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*3;int tw=((tq->u-tq->l+1)-5+3)>>2;float const*tx=tr;__m128 ty
,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tx));(ub)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ub)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+4)));(ub)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+8)));;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=3*4;__asm__(""::"r"(tv));(uc
)=_mm_loadu_ps((float const*)(tx+(0)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3))));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+4))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+8)
)));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ub)=_mm_load_ss((float const*)(tx+
(4)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2
)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(
tv+(4)*3))));;(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(////////
_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6))
);(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ua,tz
,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ua)=/////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(ty)=_mm_add_ps(ty,uc);(ub)=_mm_add_ps(ub,ua);(ty)=_mm_add_ps(ty,ub);tr
+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),ty);continue;}(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(ty));_mm_store_ss((//
float*)(tu+2-3),tz);break;;}while(tu<tt);}static void ox(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*3;int tw=((tq->u-tq->l+1)-6+3)>>2;float const*tx=tr;__m128 ty
,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tx));(ub)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1
<<6)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ub)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+4)));(ub)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+8)));;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=3*4;__asm__(""::"r"(tv));(uc
)=_mm_loadu_ps((float const*)(tx+(0)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3))));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+4))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+8)
)));;--tw;}while(tw>0);{__m128 ud;__asm__(""::"r"(tv));(uc)=_mm_castsi128_ps(//
_mm_loadl_epi64((__m128i*)(tx+(4))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*3))));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(ud)=/////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tv+(4)*3+4)));(tz)=_mm_add_ps(tz,//
_mm_mul_ps(ub,ud));};(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(
_mm_shuffle_ps(tz,ty,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6))
);(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ua,tz
,(0<<0)+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ua)=/////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(2<<2)+(3<<4)+(0
<<6)));(ty)=_mm_add_ps(ty,uc);(ub)=_mm_add_ps(ub,ua);(ty)=_mm_add_ps(ty,ub);tr
+=ts;++tq;tu+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),ty);continue;}(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1
<<6)));_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(ty));_mm_store_ss((//
float*)(tu+2-3),tz);break;;}while(tu<tt);}static void oy(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*3;///
float*__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{////
float const*tv=tp+tq->l*3;int tw=((tq->u-tq->l+1)-7+3)>>2;float const*tx=tr;///
__m128 ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tx)
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0
<<4)+(1<<6)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ub)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2
<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+4)));(ub)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+8)));;_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=3*4;__asm__(""::"r"(tv));(uc
)=_mm_loadu_ps((float const*)(tx+(0)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3))));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+4))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(0)*3+8)
)));;--tw;}while(tw>0);{__m128 ud;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float
const*)(tx+(4)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(
0<<0)+(0<<2)+(0<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((///
float const*)(tv+(4)*3))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(////////////
_mm_castps_si128(uc),(1<<0)+(1<<2)+(2<<4)+(2<<6)));(tz)=_mm_add_ps(tz,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*3+4))));(ub)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ud)=/////
_mm_load_ss((float const*)(tv+(4)*3+8));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,ud));}
;(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(tz,ty,
(0<<0)+(1<<2)+(2<<4)+(3<<6))),(3<<0)+(0<<2)+(1<<4)+(2<<6)));(uc)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_shuffle_ps(ua,tz,(0<<0)
+(1<<2)+(2<<4)+(3<<6))),(2<<0)+(3<<2)+(0<<4)+(1<<6)));(ua)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(2<<2)+(3<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,uc);(ub)=_mm_add_ps(ub,ua);(ty)=_mm_add_ps(ty,ub);tr+=ts;++tq;tu
+=3;if(tu<tt){_mm_storeu_ps((float*)(tu-3),ty);continue;}(tz)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ty),(2<<0)+(3<<2)+(0<<4)+(1<<6)));//////////
_mm_storel_epi64((__m128i*)(tu-3),_mm_castps_si128(ty));_mm_store_ss((float*)(
tu+2-3),tz);break;;}while(tu<tt);}static gn*oz[4]={ov,ow,ox,oy,};static gn*pa[
12]={oi,oj,ol,om,on,oo,op,oq,or,os,ot,ou,};static void pb(float*tn,unsigned int
to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*4;///
float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float
const*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty;__asm__(""::"r"(tv));(ty)=//
_mm_load_ss((float const*)(tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(///////
_mm_castps_si128(ty),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv)));;_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;
tu+=4;;}while(tu<tt);}static void pd(float*tn,unsigned int to,float const*tp,ge
const*tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn
;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;///
float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(tz)=_mm_castsi128_ps(//
_mm_loadl_epi64((__m128i*)(tw)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(tz),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tz),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv+4))));;_mm_storeu_ps((float*)(tu),
tx);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pe(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*4;float*
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(tz)=////
_mm_loadu_ps((float const*)(tw));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(tz),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ty,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(tz),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv+4))));(ty)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(tz),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ty,_mm_loadu_ps((float const*)(tv+8))));;_mm_storeu_ps
((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pf(float*tn,///
unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){float const*
tt=tn+to*4;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")do{float const*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty,tz,
ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}
while(tu<tt);}static void pg(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn;/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;////
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(tz)=_mm_load_ss((float const*)(tw+(4)));(tz)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}//
while(tu<tt);}static void ph(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn;/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;////
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(4
))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)
+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+(4)*4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0
)+(1<<2)+(1<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float//
const*)(tv+(4)*4+4))));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);
tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pi(float*tn,unsigned int to,float
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*4;float*//////
__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float/////
const*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(
ua)=_mm_loadu_ps((float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,/////////
_mm_loadu_ps((float const*)(tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(/////
_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,/////////
_mm_loadu_ps((float const*)(tv+4)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(tz)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))));;__asm__(""::
"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4))));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4+4)
)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+
(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv
+(4)*4+8))));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq
;tu+=4;;}while(tu<tt);}static void pj(float*tn,unsigned int to,float const*tp,
ge const*tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu
=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1
<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*4+4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float////
const*)(tv+(4)*4+8))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,///////////
_mm_loadu_ps((float const*)(tv+(4)*4+12))));;(tx)=_mm_add_ps(tx,ty);///////////
_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pk
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*4;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty,tz,
ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1
<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*4+4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float////
const*)(tv+(4)*4+8))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,///////////
_mm_loadu_ps((float const*)(tv+(4)*4+12))));;__asm__(""::"r"(tv));(tz)=////////
_mm_load_ss((float const*)(tw+(8)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(tz),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4))));;(tx)=_mm_add_ps(tx,ty);
_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pl
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*4;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*4;float const*tw=tr;__m128 tx,ty,tz,
ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(tz)=//////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=////////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1
<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*4+4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float////
const*)(tv+(4)*4+8))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,///////////
_mm_loadu_ps((float const*)(tv+(4)*4+12))));;__asm__(""::"r"(tv));(ua)=////////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tw+(8))));(tz)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4))));(tz)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4+4)
)));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}
while(tu<tt);}static void pm(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn;;////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;////
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1
<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*4+4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float////
const*)(tv+(4)*4+8))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,///////////
_mm_loadu_ps((float const*)(tv+(4)*4+12))));;__asm__(""::"r"(tv));(ua)=////////
_mm_loadu_ps((float const*)(tw+(8)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4))));(tz)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4+4))));(tz)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4+8)
)));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=4;;}
while(tu<tt);}static void pn(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn;/////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;////
float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((//
float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+4)));(tz)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+8))));(
tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+12))
));;__asm__(""::"r"(tv));(ua)=_mm_loadu_ps((float const*)(tw+(4)));(tz)=///////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(4)*4)))
);(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1
<<4)+(1<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(
4)*4+4))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+
(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float////
const*)(tv+(4)*4+8))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ua),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,///////////
_mm_loadu_ps((float const*)(tv+(4)*4+12))));;__asm__(""::"r"(tv));(ua)=////////
_mm_loadu_ps((float const*)(tw+(8)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4))));(tz)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4+4))));(tz)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+(8)*4+8)
)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(3<<0)+(3<<2)+
(3<<4)+(3<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv
+(8)*4+12))));;(tx)=_mm_add_ps(tx,ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++
tq;tu+=4;;}while(tu<tt);}static void po(float*tn,unsigned int to,float const*tp
,ge const*tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__//
tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*
4;int tw=((tq->u-tq->l+1)-4+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(
""::"r"(tv));(ub)=_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(/////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(///////
_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8))));(ua)=/////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+12))));;
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=4*4;__asm__(""::"r"(
tv));(ub)=_mm_loadu_ps((float const*)(tx+(0)));(ua)=_mm_castsi128_ps(//////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4+4)
)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+
(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv
+(0)*4+8))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0
)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float//
const*)(tv+(0)*4+12))));;--tw;}while(tw>0);(ty)=_mm_add_ps(ty,tz);_mm_storeu_ps
((float*)(tu),ty);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pp(float*tn,///
unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){float const*
tt=tn+to*4;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(////////////
"GCC novector")do{float const*tv=tp+tq->l*4;int tw=((tq->u-tq->l+1)-5+3)>>2;///
float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tx));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(
tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8))));(
ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+12))
));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=4*4;__asm__(""::
"r"(tv));(ub)=_mm_loadu_ps((float const*)(tx+(0)));(ua)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4+4)
)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+
(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv
+(0)*4+8))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0
)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float//
const*)(tv+(0)*4+12))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ua)=//////////
_mm_load_ss((float const*)(tx+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4))));;(ty)=_mm_add_ps(ty,tz);
_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu+=4;;}while(tu<tt);}static void pq
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*4;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*4;int tw=((tq->u-tq->l+1)-6+3)>>2;///
float const*tx=tr;__m128 ty,tz,ua,ub;__asm__(""::"r"(tv));(ub)=_mm_loadu_ps((//
float const*)(tx));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub)
,(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(
tv)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2
)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8))));(
ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0)+(3<<2)+(3<<4
)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+12))
));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=4*4;__asm__(""::
"r"(tv));(ub)=_mm_loadu_ps((float const*)(tx+(0)));(ua)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4+4)
)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+
(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv
+(0)*4+8))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0
)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float//
const*)(tv+(0)*4+12))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ub)=//////////
_mm_castsi128_ps(_mm_loadl_epi64((__m128i*)(tx+(4))));(ua)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4+4)
)));;(ty)=_mm_add_ps(ty,tz);_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu+=4;;}
while(tu<tt);}static void pr(float*tn,unsigned int to,float const*tp,ge const*
tq,float const*tr,int ts){float const*tt=tn+to*4;float*__restrict__ tu=tn;;////
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*4;int
tw=((tq->u-tq->l+1)-7+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub;__asm__("":://
"r"(tv));(ub)=_mm_loadu_ps((float const*)(tx));(ua)=_mm_castsi128_ps(//////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv)));(ua)=_mm_castsi128_ps(/////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+4)));(ua)=_mm_castsi128_ps(///////
_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+8))));(ua)=/////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+12))));;
_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=4*4;__asm__(""::"r"(
tv));(ub)=_mm_loadu_ps((float const*)(tx+(0)));(ua)=_mm_castsi128_ps(//////////
_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4))));(ua)=/////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(0)*4+4)
)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+
(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv
+(0)*4+8))));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(3<<0
)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float//
const*)(tv+(0)*4+12))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ub)=//////////
_mm_loadu_ps((float const*)(tx+(4)));(ua)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ub),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4))));(ua)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ub),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4+4))));(ua)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(ua,_mm_loadu_ps((float const*)(tv+(4)*4+8)
)));;(ty)=_mm_add_ps(ty,tz);_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu+=4;;}
while(tu<tt);}static gn*ps[4]={po,pp,pq,pr,};static gn*pt[12]={pb,pd,pe,pf,pg,
ph,pi,pj,pk,pl,pm,pn,};static void pu(float*tn,unsigned int to,float const*tp,
ge const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu
=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;
float const*tw=tr;__m128 tx,ty,tz;__asm__(""::"r"(tv));(tz)=_mm_load_ss((float
const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(tz),(0<<0
)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv)));(
ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*)(tv+3)));;_mm_storeu_ps((float*)(
tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}////
static void pv(float*tn,unsigned int to,float const*tp,ge const*tq,float const*
tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma(////////////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;float const*
tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=_mm_castsi128_ps(///////////
_mm_loadl_epi64((__m128i*)(tw)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*
)(tv+3)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float/////
const*)(tv+7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+10))));;_mm_storeu_ps((float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=
ts;++tq;tu+=7;;}while(tu<tt);}static void pw(float*tn,unsigned int to,float////
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*//////
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*7;float const*tw=tr;__m128 tx,ty,tz,ua;__asm__(""::"r"(tv));(ua)=
_mm_loadu_ps((float const*)(tw));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ua),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(tz,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(tz,_mm_loadu_ps((float const*
)(tv+3)));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float/////
const*)(tv+7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+10))));(tz)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ua),(2<<0)+(
2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(tz,_mm_loadu_ps((float/////
const*)(tv+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(tz,_mm_loadu_ps((float const*)(
tv+17))));;_mm_storeu_ps((float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=
ts;++tq;tu+=7;;}while(tu<tt);}static void px(float*tn,unsigned int to,float////
const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*//////
__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const
*tv=tp+tq->l*7;float const*tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));
(uc)=_mm_loadu_ps((float const*)(tw));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*
)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+7)));(
ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+10)));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+14))));(ty)=////////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+17))));(ub)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+21))));(
ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+24))));;(tx)=///
_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((float*)(tu+3),ty);/////
_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void py
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*7;float const*tw=tr;__m128 tx,ty,tz,
ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw));(ub)=//////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,
_mm_loadu_ps((float const*)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(
2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((///
float const*)(tv+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),
(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((//
float const*)(tv+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+24))));;__asm__(""::"r"(tv));(ub)=_mm_load_ss((float const*)(tw+(4))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0<<2)+(0
<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
4)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+
3))));;(tx)=_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((float*)(tu+
3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static
void pz(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int
ts){float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")////
_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;float const*tw=tr;__m128 tx
,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw));(ub)
=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(
0<<6)));(tx)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub
,_mm_loadu_ps((float const*)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(
2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((///
float const*)(tv+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),
(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((//
float const*)(tv+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+24))));;__asm__(""::"r"(tv));(uc)=_mm_castsi128_ps(_mm_loadl_epi64((
__m128i*)(tw+(4))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc
),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((
float const*)(tv+(4)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float
const*)(tv+(4)*7+3))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,///////////
_mm_loadu_ps((float const*)(tv+(4)*7+7))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,///
_mm_loadu_ps((float const*)(tv+(4)*7+10))));;(tx)=_mm_add_ps(tx,tz);(ty)=//////
_mm_add_ps(ty,ua);_mm_storeu_ps((float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),
tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void qa(float*tn,unsigned int to,
float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*
__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float/////
const*tv=tp+tq->l*7;float const*tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(
tv));(uc)=_mm_loadu_ps((float const*)(tw));(ub)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps
((float const*)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(
2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((///
float const*)(tv+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),
(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((//
float const*)(tv+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+24))));;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)
));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(
0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+
(4)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7
+3))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<
2)+(1<<4)+(1<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)
(tv+(4)*7+7))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv
+(4)*7+10))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<
0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float
const*)(tv+(4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+17))));;(tx)=_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);////////
_mm_storeu_ps((float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu
+=7;;}while(tu<tt);}static void qb(float*tn,unsigned int to,float const*tp,ge//
const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn
;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;///
float const*tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=//////////
_mm_loadu_ps((float const*)(tw));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*
)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+7)));(
ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+10)));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+14))));(ty)=////////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+17))));(ub)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+21))));(
ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+24))));;__asm__(
""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7))));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+3))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+7)
)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+10)))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2
<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)
*7+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(
3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+(4)*7+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+24))));;(tx)=_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);////////
_mm_storeu_ps((float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu
+=7;;}while(tu<tt);}static void qc(float*tn,unsigned int to,float const*tp,ge//
const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn
;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;///
float const*tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=//////////
_mm_loadu_ps((float const*)(tw));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*
)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+7)));(
ua)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+10)));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+14))));(ty)=////////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+17))));(ub)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+21))));(
ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+24))));;__asm__(
""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)));(ub)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7))));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+3))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+7)
)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+10)))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2
<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)
*7+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(
3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+(4)*7+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+24))));;__asm__(""::"r"(tv));(ub)=_mm_load_ss((float const*)(
tw+(8)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ub),(0<<0)+(0
<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const
*)(tv+(8)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv
+(8)*7+3))));;(tx)=_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((////
float*)(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<
tt);}static void qd(float*tn,unsigned int to,float const*tp,ge const*tq,float//
const*tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;float const*
tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float///
const*)(tw));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0
)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(
ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+3)));(ub)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,//////////
_mm_loadu_ps((float const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+14))));(ty)=_mm_add_ps(ty,////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+17))));(ub)=_mm_castsi128_ps(/////
_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+21))));(ua)=////////
_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+24))));;__asm__(""::
"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)));(ub)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7))));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+3))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+7)
)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+10)))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2
<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)
*7+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(
3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+(4)*7+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+24))));;__asm__(""::"r"(tv));(uc)=_mm_castsi128_ps(///////////
_mm_loadl_epi64((__m128i*)(tw+(8))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7))));(ty)=_mm_add_ps(ty,/////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+3))));(ub)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+7))));(ua)=///
_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+10))));;(tx)=
_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((float*)(tu+3),ty);/////
_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void qe
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*7;float*__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma
("GCC novector")do{float const*tv=tp+tq->l*7;float const*tw=tr;__m128 tx,ty,tz,
ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw));(ub)=//////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(tx)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(ty)=_mm_mul_ps(ub,
_mm_loadu_ps((float const*)(tv+3)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_mul_ps(ub,/////////
_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(
2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((///
float const*)(tv+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),
(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((//
float const*)(tv+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+24))));;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)
));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(
0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+
(4)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7
+3))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<
2)+(1<<4)+(1<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)
(tv+(4)*7+7))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv
+(4)*7+10))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<
0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float
const*)(tv+(4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+21))));(ua)=_mm_add_ps(ua,//
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+24))));;__asm__(""::"r"(tv))
;(uc)=_mm_loadu_ps((float const*)(tw+(8)));(ub)=_mm_castsi128_ps(//////////////
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7))));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+3))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+7)
)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+10)))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2
<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
8)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)
*7+17))));;(tx)=_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((float*)
(tu+3),ty);_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}///
static void qf(float*tn,unsigned int to,float const*tp,ge const*tq,float const*
tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma(////////////
"GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;float const*
tw=tr;__m128 tx,ty,tz,ua,ub,uc;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float///
const*)(tw));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0
)+(0<<2)+(0<<4)+(0<<6)));(tx)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv)));(
ty)=_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+3)));(ub)=_mm_castsi128_ps(///
_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(tz)=/////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+7)));(ua)=_mm_mul_ps(ub,//////////
_mm_loadu_ps((float const*)(tv+10)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+14))));(ty)=_mm_add_ps(ty,////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+17))));(ub)=_mm_castsi128_ps(/////
_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=/////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+21))));(ua)=////////
_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+24))));;__asm__(""::
"r"(tv));(uc)=_mm_loadu_ps((float const*)(tw+(4)));(ub)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(tx)=/////
_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7))));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+3))));(ub)=///
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0)+(1<<2)+(1<<4)+(1
<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+7)
)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)*7+10)))
);(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(2<<0)+(2<<2)+(2
<<4)+(2<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(
4)*7+14))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(4)
*7+17))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(
3<<2)+(3<<4)+(3<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float/////
const*)(tv+(4)*7+21))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float///
const*)(tv+(4)*7+24))));;__asm__(""::"r"(tv));(uc)=_mm_loadu_ps((float const*)(
tw+(8)));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(0<<0)+(0
<<2)+(0<<4)+(0<<6)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,_mm_loadu_ps((float const
*)(tv+(8)*7))));(ty)=_mm_add_ps(ty,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv
+(8)*7+3))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(uc),(1<<0
)+(1<<2)+(1<<4)+(1<<6)));(tz)=_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float//
const*)(tv+(8)*7+7))));(ua)=_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float////
const*)(tv+(8)*7+10))));(ub)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(uc),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(tx)=_mm_add_ps(tx,/////////
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+14))));(ty)=_mm_add_ps(ty,//
_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+17))));(ub)=_mm_castsi128_ps
(_mm_shuffle_epi32(_mm_castps_si128(uc),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(tz)=////
_mm_add_ps(tz,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+21))));(ua)=//
_mm_add_ps(ua,_mm_mul_ps(ub,_mm_loadu_ps((float const*)(tv+(8)*7+24))));;(tx)=
_mm_add_ps(tx,tz);(ty)=_mm_add_ps(ty,ua);_mm_storeu_ps((float*)(tu+3),ty);/////
_mm_storeu_ps((float*)(tu),tx);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void qg
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*7;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma(
"GCC novector")do{float const*tv=tp+tq->l*7;int tw=((tq->u-tq->l+1)-4+3)>>2;///
float const*tx=tr;__m128 ty,tz,ua,ub,uc,ud;__asm__(""::"r"(tv));(ud)=//////////
_mm_loadu_ps((float const*)(tx));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_mul_ps(uc,/////////
_mm_loadu_ps((float const*)(tv)));(tz)=_mm_mul_ps(uc,_mm_loadu_ps((float const*
)(tv+3)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(ua)=_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+7)));(
ub)=_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+10)));(uc)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+14))));(tz)=////////
_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+17))));(uc)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+21))));(
ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+24))));;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=7*4;__asm__(""::"r"(tv));(ud
)=_mm_loadu_ps((float const*)(tx+(0)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7))));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+3))));(uc)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+7))));(ub)=///
_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+10))));(uc)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+14
))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+17))
));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(
3<<4)+(3<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+
(0)*7+21))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0
)*7+24))));;--tw;}while(tw>0);(ty)=_mm_add_ps(ty,ua);(tz)=_mm_add_ps(tz,ub);///
_mm_storeu_ps((float*)(tu+3),tz);_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu
+=7;;}while(tu<tt);}static void qh(float*tn,unsigned int to,float const*tp,ge//
const*tq,float const*tr,int ts){float const*tt=tn+to*7;float*__restrict__ tu=tn
;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{float const*tv=tp+tq->l*7;int
tw=((tq->u-tq->l+1)-5+3)>>2;float const*tx=tr;__m128 ty,tz,ua,ub,uc,ud;__asm__(
""::"r"(tv));(ud)=_mm_loadu_ps((float const*)(tx));(uc)=_mm_castsi128_ps(//////
_mm_shuffle_epi32(_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv)));(tz)=_mm_mul_ps(uc,_mm_loadu_ps
((float const*)(tv+3)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(///////////////
_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=_mm_mul_ps(uc,/////////
_mm_loadu_ps((float const*)(tv+7)));(ub)=_mm_mul_ps(uc,_mm_loadu_ps((float/////
const*)(tv+10)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(
2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((///
float const*)(tv+14))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float///
const*)(tv+17))));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),
(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((//
float const*)(tv+21))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float///
const*)(tv+24))));;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=7
*4;__asm__(""::"r"(tv));(ud)=_mm_loadu_ps((float const*)(tx+(0)));(uc)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7)))
);(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+3))));(
uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4
)+(1<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*
7+7))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+
10))));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<
2)+(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)
(tv+(0)*7+14))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(
tv+(0)*7+17))));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3
<<0)+(3<<2)+(3<<4)+(3<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((////
float const*)(tv+(0)*7+21))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((//
float const*)(tv+(0)*7+24))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(uc)=////
_mm_load_ss((float const*)(tx+(4)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(///
_mm_castps_si128(uc),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7))));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+3))));;(ty)=_mm_add_ps(ty,ua
);(tz)=_mm_add_ps(tz,ub);_mm_storeu_ps((float*)(tu+3),tz);_mm_storeu_ps((float*
)(tu),ty);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void qi(float*tn,unsigned///
int to,float const*tp,ge const*tq,float const*tr,int ts){float const*tt=tn+to*7
;float*__restrict__ tu=tn;_Pragma("GCC unroll 1")_Pragma("GCC novector")do{////
float const*tv=tp+tq->l*7;int tw=((tq->u-tq->l+1)-6+3)>>2;float const*tx=tr;///
__m128 ty,tz,ua,ub,uc,ud;__asm__(""::"r"(tv));(ud)=_mm_loadu_ps((float const*)(
tx));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(0<<0)+(0<<2)
+(0<<4)+(0<<6)));(ty)=_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv)));(tz)=////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+3)));(uc)=_mm_castsi128_ps(///////
_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+7)));(ub)=_mm_mul_ps(uc,//////////
_mm_loadu_ps((float const*)(tv+10)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+14))));(tz)=_mm_add_ps(tz,////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+17))));(uc)=_mm_castsi128_ps(/////
_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(3<<4)+(3<<6)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+21))));(ub)=////////
_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+24))));;_Pragma(////
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=7*4;__asm__(""::"r"(tv));(ud
)=_mm_loadu_ps((float const*)(tx+(0)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7))));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+3))));(uc)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+7))));(ub)=///
_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+10))));(uc)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+14
))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+17))
));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(
3<<4)+(3<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+
(0)*7+21))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0
)*7+24))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ud)=_mm_castsi128_ps(//////
_mm_loadl_epi64((__m128i*)(tx+(4))));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(//
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7))));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+3))));(uc)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+7))));(ub)=///
_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+10))));;(ty)=
_mm_add_ps(ty,ua);(tz)=_mm_add_ps(tz,ub);_mm_storeu_ps((float*)(tu+3),tz);/////
_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static void qj
(float*tn,unsigned int to,float const*tp,ge const*tq,float const*tr,int ts){///
float const*tt=tn+to*7;float*__restrict__ tu=tn;;_Pragma("GCC unroll 1")_Pragma
("GCC novector")do{float const*tv=tp+tq->l*7;int tw=((tq->u-tq->l+1)-7+3)>>2;//
float const*tx=tr;__m128 ty,tz,ua,ub,uc,ud;__asm__(""::"r"(tv));(ud)=//////////
_mm_loadu_ps((float const*)(tx));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(//////
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_mul_ps(uc,/////////
_mm_loadu_ps((float const*)(tv)));(tz)=_mm_mul_ps(uc,_mm_loadu_ps((float const*
)(tv+3)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(
1<<2)+(1<<4)+(1<<6)));(ua)=_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+7)));(
ub)=_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+10)));(uc)=_mm_castsi128_ps(//
_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+14))));(tz)=////////
_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+17))));(uc)=////////
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(3<<4)+(3
<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+21))));(
ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+24))));;_Pragma(
"GCC unroll 1")_Pragma("GCC novector")do{tx+=4;tv+=7*4;__asm__(""::"r"(tv));(ud
)=_mm_loadu_ps((float const*)(tx+(0)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(
_mm_castps_si128(ud),(0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,/////////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7))));(tz)=_mm_add_ps(tz,/////
_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+3))));(uc)=_mm_castsi128_ps(
_mm_shuffle_epi32(_mm_castps_si128(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+7))));(ub)=///
_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+10))));(uc)=//
_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2
<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+14
))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0)*7+17))
));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(3<<0)+(3<<2)+(
3<<4)+(3<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+
(0)*7+21))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(0
)*7+24))));;--tw;}while(tw>0);__asm__(""::"r"(tv));(ud)=_mm_loadu_ps((float////
const*)(tx+(4)));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(ud),(
0<<0)+(0<<2)+(0<<4)+(0<<6)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((///
float const*)(tv+(4)*7))));(tz)=_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float
const*)(tv+(4)*7+3))));(uc)=_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128
(ud),(1<<0)+(1<<2)+(1<<4)+(1<<6)));(ua)=_mm_add_ps(ua,_mm_mul_ps(uc,///////////
_mm_loadu_ps((float const*)(tv+(4)*7+7))));(ub)=_mm_add_ps(ub,_mm_mul_ps(uc,///
_mm_loadu_ps((float const*)(tv+(4)*7+10))));(uc)=_mm_castsi128_ps(/////////////
_mm_shuffle_epi32(_mm_castps_si128(ud),(2<<0)+(2<<2)+(2<<4)+(2<<6)));(ty)=/////
_mm_add_ps(ty,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+14))));(tz)=//
_mm_add_ps(tz,_mm_mul_ps(uc,_mm_loadu_ps((float const*)(tv+(4)*7+17))));;(ty)=
_mm_add_ps(ty,ua);(tz)=_mm_add_ps(tz,ub);_mm_storeu_ps((float*)(tu+3),tz);/////
_mm_storeu_ps((float*)(tu),ty);tr+=ts;++tq;tu+=7;;}while(tu<tt);}static gn*qk[4
]={qg,qh,qi,qj,};static gn*ql[12]={pu,pv,pw,px,py,pz,qa,qb,qc,qd,qe,qf,};static
void qm(float**tn,float const*to,float const*tp,float const*tq){float*/////////
__restrict__ tr=tn[0];float ts=to[0];{__m128 tt=_mm_set_ps1(ts);_Pragma(///////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){////
__m128 tu,tv,tw,tx,ty,tz,ua,ub;__asm__(""::"r"(tr));(ty)=_mm_loadu_ps((float///
const*)(tp));(tz)=_mm_loadu_ps((float const*)(tp+4));(ua)=_mm_loadu_ps((float//
const*)(tp+(2*4)));(ub)=_mm_loadu_ps((float const*)(tp+(3*4)));(tu)=_mm_mul_ps(
ty,tt);(tv)=_mm_mul_ps(tz,tt);(tw)=_mm_mul_ps(ua,tt);(tx)=_mm_mul_ps(ub,tt);///
_mm_storeu_ps((float*)(tr),tu);_mm_storeu_ps((float*)(tr+4),tv);_mm_storeu_ps((
float*)(tr+(2*4)),tw);_mm_storeu_ps((float*)(tr+(3*4)),tx);tp+=(4*4);tr+=(4*4);
}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16)
{__m128 uc,ud;__asm__(""::"r"(tr));(ud)=_mm_loadu_ps((float const*)(tp));(uc)=
_mm_mul_ps(ud,(tt));_mm_storeu_ps((float*)(tr),uc);tp+=4;tr+=4;}}_Pragma(//////
"GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float ue=tp[0];__asm__(""::
"r"(tr));tr[0]=(ue*ts);++tp;++tr;}}static void qn(float*tn,float const*to,float
const**tp,float const*tq){float*__restrict__ tr=tn;float const*ts=tp[0];float//
tt=to[0];if((tt>=(1.0f-0.000001f))&&(tt<=(1.0f+0.000001f))){hx(tr,ts,(char*)tq-
(char*)ts);return;}{__m128 tu=_mm_set_ps1(tt);_Pragma("GCC unroll 1")_Pragma(//
"GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 tv,tw,tx,ty,tz,ua,ub
,uc;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);(tz)=////
_mm_loadu_ps((float const*)(ts));(ua)=_mm_loadu_ps((float const*)(ts+4));(ub)=
_mm_loadu_ps((float const*)(ts+(2*4)));(uc)=_mm_loadu_ps((float const*)(ts+(3*4
)));(tv)=_mm_mul_ps(tz,tu);(tw)=_mm_mul_ps(ua,tu);(tx)=_mm_mul_ps(ub,tu);(ty)=
_mm_mul_ps(uc,tu);_mm_storeu_ps((float*)(tr),tv);_mm_storeu_ps((float*)(tr+4),
tw);_mm_storeu_ps((float*)(tr+(2*4)),tx);_mm_storeu_ps((float*)(tr+(3*4)),ty);
tr+=(4*4);ts+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char
*)tq-(char*)ts)>=16){__m128 ud,ue;__asm__(""::"r"(tr));(ue)=_mm_loadu_ps((float
const*)(ts));(ud)=_mm_mul_ps(ue,(tu));_mm_storeu_ps((float*)(tr),ud);tr+=4;ts+=
4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float uf;__asm__
(""::"r"(tr));uf=ts[0]*tt;tr[0]=uf;++tr;++ts;}}static void qo(float**tn,float//
const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to
[0];{__m128 tt=_mm_set_ps1(ts);_Pragma("GCC unroll 1")_Pragma("GCC novector")//
while(((char*)tq-(char*)tp)>=(16*4)){__m128 tu,tv,tw,tx,ty,tz,ua,ub;__asm__(""
::"r"(tr));(ty)=_mm_loadu_ps((float const*)(tp));(tz)=_mm_loadu_ps((float const
*)(tp+4));(ua)=_mm_loadu_ps((float const*)(tp+(2*4)));(ub)=_mm_loadu_ps((float
const*)(tp+(3*4)));(tu)=_mm_loadu_ps((float const*)(tr));(tv)=_mm_loadu_ps((///
float const*)(tr+4));(tw)=_mm_loadu_ps((float const*)(tr+(2*4)));(tx)=/////////
_mm_loadu_ps((float const*)(tr+(3*4)));(tu)=_mm_add_ps(tu,_mm_mul_ps(ty,tt));(
tv)=_mm_add_ps(tv,_mm_mul_ps(tz,tt));(tw)=_mm_add_ps(tw,_mm_mul_ps(ua,tt));(tx)
=_mm_add_ps(tx,_mm_mul_ps(ub,tt));_mm_storeu_ps((float*)(tr),tu);_mm_storeu_ps(
(float*)(tr+4),tv);_mm_storeu_ps((float*)(tr+(2*4)),tw);_mm_storeu_ps((float*)(
tr+(3*4)),tx);tp+=(4*4);tr+=(4*4);}_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 uc,ud;__asm__(""::"r"(tr
));(ud)=_mm_loadu_ps((float const*)(tp));(uc)=_mm_loadu_ps((float const*)(tr));
(uc)=_mm_add_ps(uc,_mm_mul_ps(ud,(tt)));_mm_storeu_ps((float*)(tr),uc);tp+=4;tr
+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float ue=tp[0]
;__asm__(""::"r"(tr));tr[0]+=(ue*ts);++tp;++tr;}}static void qp(float*tn,float
const*to,float const**tp,float const*tq){float*__restrict__ tr=tn;float const*
ts=tp[0];float tt=to[0];{__m128 tu=_mm_set_ps1(tt);_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 tv,tw,tx,ty,
tz,ua,ub,uc;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);(
tv)=_mm_loadu_ps((float const*)(tr));(tw)=_mm_loadu_ps((float const*)(tr+4));(
tx)=_mm_loadu_ps((float const*)(tr+(2*4)));(ty)=_mm_loadu_ps((float const*)(tr+
(3*4)));(tz)=_mm_loadu_ps((float const*)(ts));(ua)=_mm_loadu_ps((float const*)(
ts+4));(ub)=_mm_loadu_ps((float const*)(ts+(2*4)));(uc)=_mm_loadu_ps((float////
const*)(ts+(3*4)));(tv)=_mm_add_ps(tv,_mm_mul_ps(tz,tu));(tw)=_mm_add_ps(tw,///
_mm_mul_ps(ua,tu));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,tu));(ty)=_mm_add_ps(ty,///
_mm_mul_ps(uc,tu));_mm_storeu_ps((float*)(tr),tv);_mm_storeu_ps((float*)(tr+4),
tw);_mm_storeu_ps((float*)(tr+(2*4)),tx);_mm_storeu_ps((float*)(tr+(3*4)),ty);
tr+=(4*4);ts+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char
*)tq-(char*)ts)>=16){__m128 ud,ue;__asm__(""::"r"(tr));(ud)=_mm_loadu_ps((float
const*)(tr));(ue)=_mm_loadu_ps((float const*)(ts));(ud)=_mm_add_ps(ud,/////////
_mm_mul_ps(ue,(tu)));_mm_storeu_ps((float*)(tr),ud);tr+=4;ts+=4;}}_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float uf;__asm__(""::"r"(tr)
);uf=tr[0]+ts[0]*tt;tr[0]=uf;++tr;++ts;}}static void qq(float**tn,float const*
to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to[0];//
float*__restrict__ tt=tn[1];float tu=to[1];{__m128 tv=_mm_set_ps1(ts);__m128 tw
=_mm_set_ps1(tu);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq
-(char*)tp)>=(16*4)){__m128 tx,ty,tz,ua,ub,uc,ud,ue;__asm__(""::"r"(tr));(ub)=
_mm_loadu_ps((float const*)(tp));(uc)=_mm_loadu_ps((float const*)(tp+4));(ud)=
_mm_loadu_ps((float const*)(tp+(2*4)));(ue)=_mm_loadu_ps((float const*)(tp+(3*4
)));(tx)=_mm_mul_ps(ub,tv);(ty)=_mm_mul_ps(uc,tv);(tz)=_mm_mul_ps(ud,tv);(ua)=
_mm_mul_ps(ue,tv);_mm_storeu_ps((float*)(tr),tx);_mm_storeu_ps((float*)(tr+4),
ty);_mm_storeu_ps((float*)(tr+(2*4)),tz);_mm_storeu_ps((float*)(tr+(3*4)),ua);(
tx)=_mm_mul_ps(ub,tw);(ty)=_mm_mul_ps(uc,tw);(tz)=_mm_mul_ps(ud,tw);(ua)=//////
_mm_mul_ps(ue,tw);_mm_storeu_ps((float*)(tt),tx);_mm_storeu_ps((float*)(tt+4),
ty);_mm_storeu_ps((float*)(tt+(2*4)),tz);_mm_storeu_ps((float*)(tt+(3*4)),ua);
tp+=(4*4);tr+=(4*4);tt+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")//
while(((char*)tq-(char*)tp)>=16){__m128 uf,ug;__asm__(""::"r"(tr));(ug)=///////
_mm_loadu_ps((float const*)(tp));(uf)=_mm_mul_ps(ug,(tv));_mm_storeu_ps((float*
)(tr),uf);(uf)=_mm_mul_ps(ug,(tw));_mm_storeu_ps((float*)(tt),uf);tp+=4;tr+=4;
tt+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uh=tp[
0];__asm__(""::"r"(tr));tr[0]=(uh*ts);tt[0]=(uh*tu);++tp;++tr;++tt;}}static////
void qr(float*tn,float const*to,float const**tp,float const*tq){float*/////////
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];{__m128 tw=_mm_set_ps1(tt);__m128 tx=_mm_set_ps1(tv);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){////
__m128 ty,tz,ua,ub,uc,ud,ue,uf;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16
*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);(uc)=///////////
_mm_loadu_ps((float const*)(ts));(ud)=_mm_loadu_ps((float const*)(ts+4));(ue)=
_mm_loadu_ps((float const*)(ts+(2*4)));(uf)=_mm_loadu_ps((float const*)(ts+(3*4
)));(ty)=_mm_mul_ps(uc,tw);(tz)=_mm_mul_ps(ud,tw);(ua)=_mm_mul_ps(ue,tw);(ub)=
_mm_mul_ps(uf,tw);(uc)=_mm_loadu_ps((float const*)(tu));(ud)=_mm_loadu_ps((////
float const*)(tu+4));(ue)=_mm_loadu_ps((float const*)(tu+(2*4)));(uf)=/////////
_mm_loadu_ps((float const*)(tu+(3*4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,tx));(
tz)=_mm_add_ps(tz,_mm_mul_ps(ud,tx));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,tx));(ub)
=_mm_add_ps(ub,_mm_mul_ps(uf,tx));_mm_storeu_ps((float*)(tr),ty);_mm_storeu_ps(
(float*)(tr+4),tz);_mm_storeu_ps((float*)(tr+(2*4)),ua);_mm_storeu_ps((float*)(
tr+(3*4)),ub);tr+=(4*4);ts+=(4*4);tu+=(4*4);}_Pragma("GCC unroll 1")_Pragma(///
"GCC novector")while(((char*)tq-(char*)ts)>=16){__m128 ug,uh;__asm__(""::"r"(tr
));(uh)=_mm_loadu_ps((float const*)(ts));(ug)=_mm_mul_ps(uh,(tw));(uh)=////////
_mm_loadu_ps((float const*)(tu));(ug)=_mm_add_ps(ug,_mm_mul_ps(uh,(tx)));//////
_mm_storeu_ps((float*)(tr),ug);tr+=4;ts+=4;tu+=4;}}_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(ts<tq){float ui;__asm__(""::"r"(tr));ui=ts[0]*tt;
ui+=tu[0]*tv;tr[0]=ui;++tr;++ts;++tu;}}static void qs(float**tn,float const*to,
float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to[0];float
*__restrict__ tt=tn[1];float tu=to[1];{__m128 tv=_mm_set_ps1(ts);__m128 tw=////
_mm_set_ps1(tu);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-
(char*)tp)>=(16*4)){__m128 tx,ty,tz,ua,ub,uc,ud,ue;__asm__(""::"r"(tr));(ub)=//
_mm_loadu_ps((float const*)(tp));(uc)=_mm_loadu_ps((float const*)(tp+4));(ud)=
_mm_loadu_ps((float const*)(tp+(2*4)));(ue)=_mm_loadu_ps((float const*)(tp+(3*4
)));(tx)=_mm_loadu_ps((float const*)(tr));(ty)=_mm_loadu_ps((float const*)(tr+4
));(tz)=_mm_loadu_ps((float const*)(tr+(2*4)));(ua)=_mm_loadu_ps((float const*)
(tr+(3*4)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,tv));(ty)=_mm_add_ps(ty,_mm_mul_ps
(uc,tv));(tz)=_mm_add_ps(tz,_mm_mul_ps(ud,tv));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue
,tv));_mm_storeu_ps((float*)(tr),tx);_mm_storeu_ps((float*)(tr+4),ty);/////////
_mm_storeu_ps((float*)(tr+(2*4)),tz);_mm_storeu_ps((float*)(tr+(3*4)),ua);(tx)=
_mm_loadu_ps((float const*)(tt));(ty)=_mm_loadu_ps((float const*)(tt+4));(tz)=
_mm_loadu_ps((float const*)(tt+(2*4)));(ua)=_mm_loadu_ps((float const*)(tt+(3*4
)));(tx)=_mm_add_ps(tx,_mm_mul_ps(ub,tw));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,tw))
;(tz)=_mm_add_ps(tz,_mm_mul_ps(ud,tw));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,tw));//
_mm_storeu_ps((float*)(tt),tx);_mm_storeu_ps((float*)(tt+4),ty);_mm_storeu_ps((
float*)(tt+(2*4)),tz);_mm_storeu_ps((float*)(tt+(3*4)),ua);tp+=(4*4);tr+=(4*4);
tt+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char
*)tp)>=16){__m128 uf,ug;__asm__(""::"r"(tr));(ug)=_mm_loadu_ps((float const*)(
tp));(uf)=_mm_loadu_ps((float const*)(tr));(uf)=_mm_add_ps(uf,_mm_mul_ps(ug,(tv
)));_mm_storeu_ps((float*)(tr),uf);(uf)=_mm_loadu_ps((float const*)(tt));(uf)=
_mm_add_ps(uf,_mm_mul_ps(ug,(tw)));_mm_storeu_ps((float*)(tt),uf);tp+=4;tr+=4;
tt+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uh=tp[
0];__asm__(""::"r"(tr));tr[0]+=(uh*ts);tt[0]+=(uh*tu);++tp;++tr;++tt;}}static//
void qt(float*tn,float const*to,float const**tp,float const*tq){float*/////////
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];{__m128 tw=_mm_set_ps1(tt);__m128 tx=_mm_set_ps1(tv);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){////
__m128 ty,tz,ua,ub,uc,ud,ue,uf;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16
*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);(ty)=///////////
_mm_loadu_ps((float const*)(tr));(tz)=_mm_loadu_ps((float const*)(tr+4));(ua)=
_mm_loadu_ps((float const*)(tr+(2*4)));(ub)=_mm_loadu_ps((float const*)(tr+(3*4
)));(uc)=_mm_loadu_ps((float const*)(ts));(ud)=_mm_loadu_ps((float const*)(ts+4
));(ue)=_mm_loadu_ps((float const*)(ts+(2*4)));(uf)=_mm_loadu_ps((float const*)
(ts+(3*4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,tw));(tz)=_mm_add_ps(tz,_mm_mul_ps
(ud,tw));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,tw));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf
,tw));(uc)=_mm_loadu_ps((float const*)(tu));(ud)=_mm_loadu_ps((float const*)(tu
+4));(ue)=_mm_loadu_ps((float const*)(tu+(2*4)));(uf)=_mm_loadu_ps((float const
*)(tu+(3*4)));(ty)=_mm_add_ps(ty,_mm_mul_ps(uc,tx));(tz)=_mm_add_ps(tz,////////
_mm_mul_ps(ud,tx));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,tx));(ub)=_mm_add_ps(ub,///
_mm_mul_ps(uf,tx));_mm_storeu_ps((float*)(tr),ty);_mm_storeu_ps((float*)(tr+4),
tz);_mm_storeu_ps((float*)(tr+(2*4)),ua);_mm_storeu_ps((float*)(tr+(3*4)),ub);
tr+=(4*4);ts+=(4*4);tu+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")//
while(((char*)tq-(char*)ts)>=16){__m128 ug,uh;__asm__(""::"r"(tr));(ug)=///////
_mm_loadu_ps((float const*)(tr));(uh)=_mm_loadu_ps((float const*)(ts));(ug)=///
_mm_add_ps(ug,_mm_mul_ps(uh,(tw)));(uh)=_mm_loadu_ps((float const*)(tu));(ug)=
_mm_add_ps(ug,_mm_mul_ps(uh,(tx)));_mm_storeu_ps((float*)(tr),ug);tr+=4;ts+=4;
tu+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float ui;///
__asm__(""::"r"(tr));ui=tr[0]+ts[0]*tt;ui+=tu[0]*tv;tr[0]=ui;++tr;++ts;++tu;}}
static void qu(float**tn,float const*to,float const*tp,float const*tq){float*//
__restrict__ tr=tn[0];float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1]
;float*__restrict__ tv=tn[2];float tw=to[2];{__m128 tx=_mm_set_ps1(ts);__m128//
ty=_mm_set_ps1(tu);__m128 tz=_mm_set_ps1(tw);_Pragma("GCC unroll 1")_Pragma(///
"GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){__m128 ua,ub,uc,ud,ue,uf,ug
,uh;__asm__(""::"r"(tr));(ue)=_mm_loadu_ps((float const*)(tp));(uf)=///////////
_mm_loadu_ps((float const*)(tp+4));(ug)=_mm_loadu_ps((float const*)(tp+(2*4)));
(uh)=_mm_loadu_ps((float const*)(tp+(3*4)));(ua)=_mm_mul_ps(ue,tx);(ub)=///////
_mm_mul_ps(uf,tx);(uc)=_mm_mul_ps(ug,tx);(ud)=_mm_mul_ps(uh,tx);_mm_storeu_ps((
float*)(tr),ua);_mm_storeu_ps((float*)(tr+4),ub);_mm_storeu_ps((float*)(tr+(2*4
)),uc);_mm_storeu_ps((float*)(tr+(3*4)),ud);(ua)=_mm_mul_ps(ue,ty);(ub)=///////
_mm_mul_ps(uf,ty);(uc)=_mm_mul_ps(ug,ty);(ud)=_mm_mul_ps(uh,ty);_mm_storeu_ps((
float*)(tt),ua);_mm_storeu_ps((float*)(tt+4),ub);_mm_storeu_ps((float*)(tt+(2*4
)),uc);_mm_storeu_ps((float*)(tt+(3*4)),ud);(ua)=_mm_mul_ps(ue,tz);(ub)=///////
_mm_mul_ps(uf,tz);(uc)=_mm_mul_ps(ug,tz);(ud)=_mm_mul_ps(uh,tz);_mm_storeu_ps((
float*)(tv),ua);_mm_storeu_ps((float*)(tv+4),ub);_mm_storeu_ps((float*)(tv+(2*4
)),uc);_mm_storeu_ps((float*)(tv+(3*4)),ud);tp+=(4*4);tr+=(4*4);tt+=(4*4);tv+=(
4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)
>=16){__m128 ui,uj;__asm__(""::"r"(tr));(uj)=_mm_loadu_ps((float const*)(tp));(
ui)=_mm_mul_ps(uj,(tx));_mm_storeu_ps((float*)(tr),ui);(ui)=_mm_mul_ps(uj,(ty))
;_mm_storeu_ps((float*)(tt),ui);(ui)=_mm_mul_ps(uj,(tz));_mm_storeu_ps((float*)
(tv),ui);tp+=4;tr+=4;tt+=4;tv+=4;}}_Pragma("GCC unroll 1")_Pragma(/////////////
"GCC novector")while(tp<tq){float uk=tp[0];__asm__(""::"r"(tr));tr[0]=(uk*ts);
tt[0]=(uk*tu);tv[0]=(uk*tw);++tp;++tr;++tt;++tv;}}static void qv(float*tn,float
const*to,float const**tp,float const*tq){float*__restrict__ tr=tn;float const*
ts=tp[0];float tt=to[0];float const*tu=tp[1];float tv=to[1];float const*tw=tp[2
];float tx=to[2];{__m128 ty=_mm_set_ps1(tt);__m128 tz=_mm_set_ps1(tv);__m128 ua
=_mm_set_ps1(tx);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq
-(char*)ts)>=(16*4)){__m128 ub,uc,ud,ue,uf,ug,uh,ui;__asm__(""::"r"(tr));//////
_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),_MM_HINT_T0);(uf)=_mm_loadu_ps((//
float const*)(ts));(ug)=_mm_loadu_ps((float const*)(ts+4));(uh)=_mm_loadu_ps((
float const*)(ts+(2*4)));(ui)=_mm_loadu_ps((float const*)(ts+(3*4)));(ub)=/////
_mm_mul_ps(uf,ty);(uc)=_mm_mul_ps(ug,ty);(ud)=_mm_mul_ps(uh,ty);(ue)=_mm_mul_ps
(ui,ty);(uf)=_mm_loadu_ps((float const*)(tu));(ug)=_mm_loadu_ps((float const*)(
tu+4));(uh)=_mm_loadu_ps((float const*)(tu+(2*4)));(ui)=_mm_loadu_ps((float////
const*)(tu+(3*4)));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,tz));(uc)=_mm_add_ps(uc,///
_mm_mul_ps(ug,tz));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,tz));(ue)=_mm_add_ps(ue,///
_mm_mul_ps(ui,tz));(uf)=_mm_loadu_ps((float const*)(tw));(ug)=_mm_loadu_ps((///
float const*)(tw+4));(uh)=_mm_loadu_ps((float const*)(tw+(2*4)));(ui)=/////////
_mm_loadu_ps((float const*)(tw+(3*4)));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,ua));(
uc)=_mm_add_ps(uc,_mm_mul_ps(ug,ua));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,ua));(ue)
=_mm_add_ps(ue,_mm_mul_ps(ui,ua));_mm_storeu_ps((float*)(tr),ub);_mm_storeu_ps(
(float*)(tr+4),uc);_mm_storeu_ps((float*)(tr+(2*4)),ud);_mm_storeu_ps((float*)(
tr+(3*4)),ue);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);}_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){__m128 uj,uk;__asm__(""
::"r"(tr));(uk)=_mm_loadu_ps((float const*)(ts));(uj)=_mm_mul_ps(uk,(ty));(uk)=
_mm_loadu_ps((float const*)(tu));(uj)=_mm_add_ps(uj,_mm_mul_ps(uk,(tz)));(uk)=
_mm_loadu_ps((float const*)(tw));(uj)=_mm_add_ps(uj,_mm_mul_ps(uk,(ua)));//////
_mm_storeu_ps((float*)(tr),uj);tr+=4;ts+=4;tu+=4;tw+=4;}}_Pragma("GCC unroll 1"
)_Pragma("GCC novector")while(ts<tq){float ul;__asm__(""::"r"(tr));ul=ts[0]*tt;
ul+=tu[0]*tv;ul+=tw[0]*tx;tr[0]=ul;++tr;++ts;++tu;++tw;}}static void qw(float**
tn,float const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];//
float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv
=tn[2];float tw=to[2];{__m128 tx=_mm_set_ps1(ts);__m128 ty=_mm_set_ps1(tu);////
__m128 tz=_mm_set_ps1(tw);_Pragma("GCC unroll 1")_Pragma("GCC novector")while((
(char*)tq-(char*)tp)>=(16*4)){__m128 ua,ub,uc,ud,ue,uf,ug,uh;__asm__(""::"r"(tr
));(ue)=_mm_loadu_ps((float const*)(tp));(uf)=_mm_loadu_ps((float const*)(tp+4)
);(ug)=_mm_loadu_ps((float const*)(tp+(2*4)));(uh)=_mm_loadu_ps((float const*)(
tp+(3*4)));(ua)=_mm_loadu_ps((float const*)(tr));(ub)=_mm_loadu_ps((float const
*)(tr+4));(uc)=_mm_loadu_ps((float const*)(tr+(2*4)));(ud)=_mm_loadu_ps((float
const*)(tr+(3*4)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,tx));(ub)=_mm_add_ps(ub,///
_mm_mul_ps(uf,tx));(uc)=_mm_add_ps(uc,_mm_mul_ps(ug,tx));(ud)=_mm_add_ps(ud,///
_mm_mul_ps(uh,tx));_mm_storeu_ps((float*)(tr),ua);_mm_storeu_ps((float*)(tr+4),
ub);_mm_storeu_ps((float*)(tr+(2*4)),uc);_mm_storeu_ps((float*)(tr+(3*4)),ud);(
ua)=_mm_loadu_ps((float const*)(tt));(ub)=_mm_loadu_ps((float const*)(tt+4));(
uc)=_mm_loadu_ps((float const*)(tt+(2*4)));(ud)=_mm_loadu_ps((float const*)(tt+
(3*4)));(ua)=_mm_add_ps(ua,_mm_mul_ps(ue,ty));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,
ty));(uc)=_mm_add_ps(uc,_mm_mul_ps(ug,ty));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,ty)
);_mm_storeu_ps((float*)(tt),ua);_mm_storeu_ps((float*)(tt+4),ub);_mm_storeu_ps
((float*)(tt+(2*4)),uc);_mm_storeu_ps((float*)(tt+(3*4)),ud);(ua)=_mm_loadu_ps(
(float const*)(tv));(ub)=_mm_loadu_ps((float const*)(tv+4));(uc)=_mm_loadu_ps((
float const*)(tv+(2*4)));(ud)=_mm_loadu_ps((float const*)(tv+(3*4)));(ua)=/////
_mm_add_ps(ua,_mm_mul_ps(ue,tz));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,tz));(uc)=///
_mm_add_ps(uc,_mm_mul_ps(ug,tz));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,tz));////////
_mm_storeu_ps((float*)(tv),ua);_mm_storeu_ps((float*)(tv+4),ub);_mm_storeu_ps((
float*)(tv+(2*4)),uc);_mm_storeu_ps((float*)(tv+(3*4)),ud);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char
*)tq-(char*)tp)>=16){__m128 ui,uj;__asm__(""::"r"(tr));(uj)=_mm_loadu_ps((float
const*)(tp));(ui)=_mm_loadu_ps((float const*)(tr));(ui)=_mm_add_ps(ui,/////////
_mm_mul_ps(uj,(tx)));_mm_storeu_ps((float*)(tr),ui);(ui)=_mm_loadu_ps((float///
const*)(tt));(ui)=_mm_add_ps(ui,_mm_mul_ps(uj,(ty)));_mm_storeu_ps((float*)(tt)
,ui);(ui)=_mm_loadu_ps((float const*)(tv));(ui)=_mm_add_ps(ui,_mm_mul_ps(uj,(tz
)));_mm_storeu_ps((float*)(tv),ui);tp+=4;tr+=4;tt+=4;tv+=4;}}_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uk=tp[0];__asm__(""::
"r"(tr));tr[0]+=(uk*ts);tt[0]+=(uk*tu);tv[0]+=(uk*tw);++tp;++tr;++tt;++tv;}}///
static void qx(float*tn,float const*to,float const**tp,float const*tq){float*//
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];float const*tw=tp[2];float tx=to[2];{__m128 ty=_mm_set_ps1(tt);
__m128 tz=_mm_set_ps1(tv);__m128 ua=_mm_set_ps1(tx);_Pragma("GCC unroll 1")////
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 ub,uc,ud,ue,
uf,ug,uh,ui;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);
_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),//
_MM_HINT_T0);(ub)=_mm_loadu_ps((float const*)(tr));(uc)=_mm_loadu_ps((float////
const*)(tr+4));(ud)=_mm_loadu_ps((float const*)(tr+(2*4)));(ue)=_mm_loadu_ps((
float const*)(tr+(3*4)));(uf)=_mm_loadu_ps((float const*)(ts));(ug)=///////////
_mm_loadu_ps((float const*)(ts+4));(uh)=_mm_loadu_ps((float const*)(ts+(2*4)));
(ui)=_mm_loadu_ps((float const*)(ts+(3*4)));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,ty
));(uc)=_mm_add_ps(uc,_mm_mul_ps(ug,ty));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,ty));
(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ty));(uf)=_mm_loadu_ps((float const*)(tu));(ug
)=_mm_loadu_ps((float const*)(tu+4));(uh)=_mm_loadu_ps((float const*)(tu+(2*4))
);(ui)=_mm_loadu_ps((float const*)(tu+(3*4)));(ub)=_mm_add_ps(ub,_mm_mul_ps(uf,
tz));(uc)=_mm_add_ps(uc,_mm_mul_ps(ug,tz));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,tz)
);(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,tz));(uf)=_mm_loadu_ps((float const*)(tw));(
ug)=_mm_loadu_ps((float const*)(tw+4));(uh)=_mm_loadu_ps((float const*)(tw+(2*4
)));(ui)=_mm_loadu_ps((float const*)(tw+(3*4)));(ub)=_mm_add_ps(ub,_mm_mul_ps(
uf,ua));(uc)=_mm_add_ps(uc,_mm_mul_ps(ug,ua));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,
ua));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ua));_mm_storeu_ps((float*)(tr),ub);/////
_mm_storeu_ps((float*)(tr+4),uc);_mm_storeu_ps((float*)(tr+(2*4)),ud);/////////
_mm_storeu_ps((float*)(tr+(3*4)),ue);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);}
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){
__m128 uj,uk;__asm__(""::"r"(tr));(uj)=_mm_loadu_ps((float const*)(tr));(uk)=//
_mm_loadu_ps((float const*)(ts));(uj)=_mm_add_ps(uj,_mm_mul_ps(uk,(ty)));(uk)=
_mm_loadu_ps((float const*)(tu));(uj)=_mm_add_ps(uj,_mm_mul_ps(uk,(tz)));(uk)=
_mm_loadu_ps((float const*)(tw));(uj)=_mm_add_ps(uj,_mm_mul_ps(uk,(ua)));//////
_mm_storeu_ps((float*)(tr),uj);tr+=4;ts+=4;tu+=4;tw+=4;}}_Pragma("GCC unroll 1"
)_Pragma("GCC novector")while(ts<tq){float ul;__asm__(""::"r"(tr));ul=tr[0]+ts[
0]*tt;ul+=tu[0]*tv;ul+=tw[0]*tx;tr[0]=ul;++tr;++ts;++tu;++tw;}}static void qy(
float**tn,float const*to,float const*tp,float const*tq){float*__restrict__ tr=
tn[0];float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*/////////
__restrict__ tv=tn[2];float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3]
;{__m128 tz=_mm_set_ps1(ts);__m128 ua=_mm_set_ps1(tu);__m128 ub=_mm_set_ps1(tw)
;__m128 uc=_mm_set_ps1(ty);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(
((char*)tq-(char*)tp)>=(16*4)){__m128 ud,ue,uf,ug,uh,ui,uj,uk;__asm__(""::"r"(
tr));(uh)=_mm_loadu_ps((float const*)(tp));(ui)=_mm_loadu_ps((float const*)(tp+
4));(uj)=_mm_loadu_ps((float const*)(tp+(2*4)));(uk)=_mm_loadu_ps((float const*
)(tp+(3*4)));(ud)=_mm_mul_ps(uh,tz);(ue)=_mm_mul_ps(ui,tz);(uf)=_mm_mul_ps(uj,
tz);(ug)=_mm_mul_ps(uk,tz);_mm_storeu_ps((float*)(tr),ud);_mm_storeu_ps((float*
)(tr+4),ue);_mm_storeu_ps((float*)(tr+(2*4)),uf);_mm_storeu_ps((float*)(tr+(3*4
)),ug);(ud)=_mm_mul_ps(uh,ua);(ue)=_mm_mul_ps(ui,ua);(uf)=_mm_mul_ps(uj,ua);(ug
)=_mm_mul_ps(uk,ua);_mm_storeu_ps((float*)(tt),ud);_mm_storeu_ps((float*)(tt+4)
,ue);_mm_storeu_ps((float*)(tt+(2*4)),uf);_mm_storeu_ps((float*)(tt+(3*4)),ug);
(ud)=_mm_mul_ps(uh,ub);(ue)=_mm_mul_ps(ui,ub);(uf)=_mm_mul_ps(uj,ub);(ug)=/////
_mm_mul_ps(uk,ub);_mm_storeu_ps((float*)(tv),ud);_mm_storeu_ps((float*)(tv+4),
ue);_mm_storeu_ps((float*)(tv+(2*4)),uf);_mm_storeu_ps((float*)(tv+(3*4)),ug);(
ud)=_mm_mul_ps(uh,uc);(ue)=_mm_mul_ps(ui,uc);(uf)=_mm_mul_ps(uj,uc);(ug)=//////
_mm_mul_ps(uk,uc);_mm_storeu_ps((float*)(tx),ud);_mm_storeu_ps((float*)(tx+4),
ue);_mm_storeu_ps((float*)(tx+(2*4)),uf);_mm_storeu_ps((float*)(tx+(3*4)),ug);
tp+=(4*4);tr+=(4*4);tt+=(4*4);tv+=(4*4);tx+=(4*4);}_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 ul,um;__asm__(""
::"r"(tr));(um)=_mm_loadu_ps((float const*)(tp));(ul)=_mm_mul_ps(um,(tz));/////
_mm_storeu_ps((float*)(tr),ul);(ul)=_mm_mul_ps(um,(ua));_mm_storeu_ps((float*)(
tt),ul);(ul)=_mm_mul_ps(um,(ub));_mm_storeu_ps((float*)(tv),ul);(ul)=_mm_mul_ps
(um,(uc));_mm_storeu_ps((float*)(tx),ul);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;}}//////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float un=tp[0];/////
__asm__(""::"r"(tr));tr[0]=(un*ts);tt[0]=(un*tu);tv[0]=(un*tw);tx[0]=(un*ty);++
tp;++tr;++tt;++tv;++tx;}}static void qz(float*tn,float const*to,float const**tp
,float const*tq){float*__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];
float const*tu=tp[1];float tv=to[1];float const*tw=tp[2];float tx=to[2];float//
const*ty=tp[3];float tz=to[3];{__m128 ua=_mm_set_ps1(tt);__m128 ub=_mm_set_ps1(
tv);__m128 uc=_mm_set_ps1(tx);__m128 ud=_mm_set_ps1(tz);_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 ue,uf,ug,uh,
ui,uj,uk,ul;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);
_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);(ui)=_mm_loadu_ps((//
float const*)(ts));(uj)=_mm_loadu_ps((float const*)(ts+4));(uk)=_mm_loadu_ps((
float const*)(ts+(2*4)));(ul)=_mm_loadu_ps((float const*)(ts+(3*4)));(ue)=/////
_mm_mul_ps(ui,ua);(uf)=_mm_mul_ps(uj,ua);(ug)=_mm_mul_ps(uk,ua);(uh)=_mm_mul_ps
(ul,ua);(ui)=_mm_loadu_ps((float const*)(tu));(uj)=_mm_loadu_ps((float const*)(
tu+4));(uk)=_mm_loadu_ps((float const*)(tu+(2*4)));(ul)=_mm_loadu_ps((float////
const*)(tu+(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ub));(uf)=_mm_add_ps(uf,///
_mm_mul_ps(uj,ub));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ub));(uh)=_mm_add_ps(uh,///
_mm_mul_ps(ul,ub));(ui)=_mm_loadu_ps((float const*)(tw));(uj)=_mm_loadu_ps((///
float const*)(tw+4));(uk)=_mm_loadu_ps((float const*)(tw+(2*4)));(ul)=/////////
_mm_loadu_ps((float const*)(tw+(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,uc));(
uf)=_mm_add_ps(uf,_mm_mul_ps(uj,uc));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,uc));(uh)
=_mm_add_ps(uh,_mm_mul_ps(ul,uc));(ui)=_mm_loadu_ps((float const*)(ty));(uj)=//
_mm_loadu_ps((float const*)(ty+4));(uk)=_mm_loadu_ps((float const*)(ty+(2*4)));
(ul)=_mm_loadu_ps((float const*)(ty+(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ud
));(uf)=_mm_add_ps(uf,_mm_mul_ps(uj,ud));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ud));
(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ud));_mm_storeu_ps((float*)(tr),ue);//////////
_mm_storeu_ps((float*)(tr+4),uf);_mm_storeu_ps((float*)(tr+(2*4)),ug);/////////
_mm_storeu_ps((float*)(tr+(3*4)),uh);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty
+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)
ts)>=16){__m128 um,un;__asm__(""::"r"(tr));(un)=_mm_loadu_ps((float const*)(ts)
);(um)=_mm_mul_ps(un,(ua));(un)=_mm_loadu_ps((float const*)(tu));(um)=/////////
_mm_add_ps(um,_mm_mul_ps(un,(ub)));(un)=_mm_loadu_ps((float const*)(tw));(um)=
_mm_add_ps(um,_mm_mul_ps(un,(uc)));(un)=_mm_loadu_ps((float const*)(ty));(um)=
_mm_add_ps(um,_mm_mul_ps(un,(ud)));_mm_storeu_ps((float*)(tr),um);tr+=4;ts+=4;
tu+=4;tw+=4;ty+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){
float uo;__asm__(""::"r"(tr));uo=ts[0]*tt;uo+=tu[0]*tv;uo+=tw[0]*tx;uo+=ty[0]*
tz;tr[0]=uo;++tr;++ts;++tu;++tw;++ty;}}static void ra(float**tn,float const*to,
float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to[0];float
*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv=tn[2];float tw=to[2
];float*__restrict__ tx=tn[3];float ty=to[3];{__m128 tz=_mm_set_ps1(ts);__m128
ua=_mm_set_ps1(tu);__m128 ub=_mm_set_ps1(tw);__m128 uc=_mm_set_ps1(ty);_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){////
__m128 ud,ue,uf,ug,uh,ui,uj,uk;__asm__(""::"r"(tr));(uh)=_mm_loadu_ps((float///
const*)(tp));(ui)=_mm_loadu_ps((float const*)(tp+4));(uj)=_mm_loadu_ps((float//
const*)(tp+(2*4)));(uk)=_mm_loadu_ps((float const*)(tp+(3*4)));(ud)=///////////
_mm_loadu_ps((float const*)(tr));(ue)=_mm_loadu_ps((float const*)(tr+4));(uf)=
_mm_loadu_ps((float const*)(tr+(2*4)));(ug)=_mm_loadu_ps((float const*)(tr+(3*4
)));(ud)=_mm_add_ps(ud,_mm_mul_ps(uh,tz));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,tz))
;(uf)=_mm_add_ps(uf,_mm_mul_ps(uj,tz));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,tz));//
_mm_storeu_ps((float*)(tr),ud);_mm_storeu_ps((float*)(tr+4),ue);_mm_storeu_ps((
float*)(tr+(2*4)),uf);_mm_storeu_ps((float*)(tr+(3*4)),ug);(ud)=_mm_loadu_ps((
float const*)(tt));(ue)=_mm_loadu_ps((float const*)(tt+4));(uf)=_mm_loadu_ps((
float const*)(tt+(2*4)));(ug)=_mm_loadu_ps((float const*)(tt+(3*4)));(ud)=/////
_mm_add_ps(ud,_mm_mul_ps(uh,ua));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ua));(uf)=///
_mm_add_ps(uf,_mm_mul_ps(uj,ua));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ua));////////
_mm_storeu_ps((float*)(tt),ud);_mm_storeu_ps((float*)(tt+4),ue);_mm_storeu_ps((
float*)(tt+(2*4)),uf);_mm_storeu_ps((float*)(tt+(3*4)),ug);(ud)=_mm_loadu_ps((
float const*)(tv));(ue)=_mm_loadu_ps((float const*)(tv+4));(uf)=_mm_loadu_ps((
float const*)(tv+(2*4)));(ug)=_mm_loadu_ps((float const*)(tv+(3*4)));(ud)=/////
_mm_add_ps(ud,_mm_mul_ps(uh,ub));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ub));(uf)=///
_mm_add_ps(uf,_mm_mul_ps(uj,ub));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ub));////////
_mm_storeu_ps((float*)(tv),ud);_mm_storeu_ps((float*)(tv+4),ue);_mm_storeu_ps((
float*)(tv+(2*4)),uf);_mm_storeu_ps((float*)(tv+(3*4)),ug);(ud)=_mm_loadu_ps((
float const*)(tx));(ue)=_mm_loadu_ps((float const*)(tx+4));(uf)=_mm_loadu_ps((
float const*)(tx+(2*4)));(ug)=_mm_loadu_ps((float const*)(tx+(3*4)));(ud)=/////
_mm_add_ps(ud,_mm_mul_ps(uh,uc));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,uc));(uf)=///
_mm_add_ps(uf,_mm_mul_ps(uj,uc));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,uc));////////
_mm_storeu_ps((float*)(tx),ud);_mm_storeu_ps((float*)(tx+4),ue);_mm_storeu_ps((
float*)(tx+(2*4)),uf);_mm_storeu_ps((float*)(tx+(3*4)),ug);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")//
while(((char*)tq-(char*)tp)>=16){__m128 ul,um;__asm__(""::"r"(tr));(um)=///////
_mm_loadu_ps((float const*)(tp));(ul)=_mm_loadu_ps((float const*)(tr));(ul)=///
_mm_add_ps(ul,_mm_mul_ps(um,(tz)));_mm_storeu_ps((float*)(tr),ul);(ul)=////////
_mm_loadu_ps((float const*)(tt));(ul)=_mm_add_ps(ul,_mm_mul_ps(um,(ua)));//////
_mm_storeu_ps((float*)(tt),ul);(ul)=_mm_loadu_ps((float const*)(tv));(ul)=/////
_mm_add_ps(ul,_mm_mul_ps(um,(ub)));_mm_storeu_ps((float*)(tv),ul);(ul)=////////
_mm_loadu_ps((float const*)(tx));(ul)=_mm_add_ps(ul,_mm_mul_ps(um,(uc)));//////
_mm_storeu_ps((float*)(tx),ul);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;}}_Pragma(////////
"GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float un=tp[0];__asm__(""::
"r"(tr));tr[0]+=(un*ts);tt[0]+=(un*tu);tv[0]+=(un*tw);tx[0]+=(un*ty);++tp;++tr;
++tt;++tv;++tx;}}static void rb(float*tn,float const*to,float const**tp,float//
const*tq){float*__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float///
const*tu=tp[1];float tv=to[1];float const*tw=tp[2];float tx=to[2];float const*
ty=tp[3];float tz=to[3];{__m128 ua=_mm_set_ps1(tt);__m128 ub=_mm_set_ps1(tv);//
__m128 uc=_mm_set_ps1(tx);__m128 ud=_mm_set_ps1(tz);_Pragma("GCC unroll 1")////
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 ue,uf,ug,uh,
ui,uj,uk,ul;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);
_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);(ue)=_mm_loadu_ps((//
float const*)(tr));(uf)=_mm_loadu_ps((float const*)(tr+4));(ug)=_mm_loadu_ps((
float const*)(tr+(2*4)));(uh)=_mm_loadu_ps((float const*)(tr+(3*4)));(ui)=/////
_mm_loadu_ps((float const*)(ts));(uj)=_mm_loadu_ps((float const*)(ts+4));(uk)=
_mm_loadu_ps((float const*)(ts+(2*4)));(ul)=_mm_loadu_ps((float const*)(ts+(3*4
)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ua));(uf)=_mm_add_ps(uf,_mm_mul_ps(uj,ua))
;(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ua));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ua));(
ui)=_mm_loadu_ps((float const*)(tu));(uj)=_mm_loadu_ps((float const*)(tu+4));(
uk)=_mm_loadu_ps((float const*)(tu+(2*4)));(ul)=_mm_loadu_ps((float const*)(tu+
(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ub));(uf)=_mm_add_ps(uf,_mm_mul_ps(uj,
ub));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ub));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ub)
);(ui)=_mm_loadu_ps((float const*)(tw));(uj)=_mm_loadu_ps((float const*)(tw+4))
;(uk)=_mm_loadu_ps((float const*)(tw+(2*4)));(ul)=_mm_loadu_ps((float const*)(
tw+(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,uc));(uf)=_mm_add_ps(uf,_mm_mul_ps(
uj,uc));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,uc));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,
uc));(ui)=_mm_loadu_ps((float const*)(ty));(uj)=_mm_loadu_ps((float const*)(ty+
4));(uk)=_mm_loadu_ps((float const*)(ty+(2*4)));(ul)=_mm_loadu_ps((float const*
)(ty+(3*4)));(ue)=_mm_add_ps(ue,_mm_mul_ps(ui,ud));(uf)=_mm_add_ps(uf,/////////
_mm_mul_ps(uj,ud));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ud));(uh)=_mm_add_ps(uh,///
_mm_mul_ps(ul,ud));_mm_storeu_ps((float*)(tr),ue);_mm_storeu_ps((float*)(tr+4),
uf);_mm_storeu_ps((float*)(tr+(2*4)),ug);_mm_storeu_ps((float*)(tr+(3*4)),uh);
tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);}_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){__m128 um,un;__asm__(""
::"r"(tr));(um)=_mm_loadu_ps((float const*)(tr));(un)=_mm_loadu_ps((float const
*)(ts));(um)=_mm_add_ps(um,_mm_mul_ps(un,(ua)));(un)=_mm_loadu_ps((float const*
)(tu));(um)=_mm_add_ps(um,_mm_mul_ps(un,(ub)));(un)=_mm_loadu_ps((float const*)
(tw));(um)=_mm_add_ps(um,_mm_mul_ps(un,(uc)));(un)=_mm_loadu_ps((float const*)(
ty));(um)=_mm_add_ps(um,_mm_mul_ps(un,(ud)));_mm_storeu_ps((float*)(tr),um);tr
+=4;ts+=4;tu+=4;tw+=4;ty+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")///
while(ts<tq){float uo;__asm__(""::"r"(tr));uo=tr[0]+ts[0]*tt;uo+=tu[0]*tv;uo+=
tw[0]*tx;uo+=ty[0]*tz;tr[0]=uo;++tr;++ts;++tu;++tw;++ty;}}static void rc(float*
*tn,float const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];
float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv
=tn[2];float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3];float*////////
__restrict__ tz=tn[4];float ua=to[4];{__m128 ub=_mm_set_ps1(ts);__m128 uc=/////
_mm_set_ps1(tu);__m128 ud=_mm_set_ps1(tw);__m128 ue=_mm_set_ps1(ty);__m128 uf=
_mm_set_ps1(ua);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-
(char*)tp)>=(16*4)){__m128 ug,uh,ui,uj,uk,ul,um,un;__asm__(""::"r"(tr));(uk)=//
_mm_loadu_ps((float const*)(tp));(ul)=_mm_loadu_ps((float const*)(tp+4));(um)=
_mm_loadu_ps((float const*)(tp+(2*4)));(un)=_mm_loadu_ps((float const*)(tp+(3*4
)));(ug)=_mm_mul_ps(uk,ub);(uh)=_mm_mul_ps(ul,ub);(ui)=_mm_mul_ps(um,ub);(uj)=
_mm_mul_ps(un,ub);_mm_storeu_ps((float*)(tr),ug);_mm_storeu_ps((float*)(tr+4),
uh);_mm_storeu_ps((float*)(tr+(2*4)),ui);_mm_storeu_ps((float*)(tr+(3*4)),uj);(
ug)=_mm_mul_ps(uk,uc);(uh)=_mm_mul_ps(ul,uc);(ui)=_mm_mul_ps(um,uc);(uj)=//////
_mm_mul_ps(un,uc);_mm_storeu_ps((float*)(tt),ug);_mm_storeu_ps((float*)(tt+4),
uh);_mm_storeu_ps((float*)(tt+(2*4)),ui);_mm_storeu_ps((float*)(tt+(3*4)),uj);(
ug)=_mm_mul_ps(uk,ud);(uh)=_mm_mul_ps(ul,ud);(ui)=_mm_mul_ps(um,ud);(uj)=//////
_mm_mul_ps(un,ud);_mm_storeu_ps((float*)(tv),ug);_mm_storeu_ps((float*)(tv+4),
uh);_mm_storeu_ps((float*)(tv+(2*4)),ui);_mm_storeu_ps((float*)(tv+(3*4)),uj);(
ug)=_mm_mul_ps(uk,ue);(uh)=_mm_mul_ps(ul,ue);(ui)=_mm_mul_ps(um,ue);(uj)=//////
_mm_mul_ps(un,ue);_mm_storeu_ps((float*)(tx),ug);_mm_storeu_ps((float*)(tx+4),
uh);_mm_storeu_ps((float*)(tx+(2*4)),ui);_mm_storeu_ps((float*)(tx+(3*4)),uj);(
ug)=_mm_mul_ps(uk,uf);(uh)=_mm_mul_ps(ul,uf);(ui)=_mm_mul_ps(um,uf);(uj)=//////
_mm_mul_ps(un,uf);_mm_storeu_ps((float*)(tz),ug);_mm_storeu_ps((float*)(tz+4),
uh);_mm_storeu_ps((float*)(tz+(2*4)),ui);_mm_storeu_ps((float*)(tz+(3*4)),uj);
tp+=(4*4);tr+=(4*4);tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);}_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128//
uo,up;__asm__(""::"r"(tr));(up)=_mm_loadu_ps((float const*)(tp));(uo)=/////////
_mm_mul_ps(up,(ub));_mm_storeu_ps((float*)(tr),uo);(uo)=_mm_mul_ps(up,(uc));///
_mm_storeu_ps((float*)(tt),uo);(uo)=_mm_mul_ps(up,(ud));_mm_storeu_ps((float*)(
tv),uo);(uo)=_mm_mul_ps(up,(ue));_mm_storeu_ps((float*)(tx),uo);(uo)=_mm_mul_ps
(up,(uf));_mm_storeu_ps((float*)(tz),uo);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;}}
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uq=tp[0];/////
__asm__(""::"r"(tr));tr[0]=(uq*ts);tt[0]=(uq*tu);tv[0]=(uq*tw);tx[0]=(uq*ty);tz
[0]=(uq*ua);++tp;++tr;++tt;++tv;++tx;++tz;}}static void rd(float*tn,float const
*to,float const**tp,float const*tq){float*__restrict__ tr=tn;float const*ts=tp[
0];float tt=to[0];float const*tu=tp[1];float tv=to[1];float const*tw=tp[2];////
float tx=to[2];float const*ty=tp[3];float tz=to[3];float const*ua=tp[4];float//
ub=to[4];{__m128 uc=_mm_set_ps1(tt);__m128 ud=_mm_set_ps1(tv);__m128 ue=///////
_mm_set_ps1(tx);__m128 uf=_mm_set_ps1(tz);__m128 ug=_mm_set_ps1(ub);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){////
__m128 uh,ui,uj,uk,ul,um,un,uo;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16
*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((//
char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);///
_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);(ul)=_mm_loadu_ps((float const*)(
ts));(um)=_mm_loadu_ps((float const*)(ts+4));(un)=_mm_loadu_ps((float const*)(
ts+(2*4)));(uo)=_mm_loadu_ps((float const*)(ts+(3*4)));(uh)=_mm_mul_ps(ul,uc);(
ui)=_mm_mul_ps(um,uc);(uj)=_mm_mul_ps(un,uc);(uk)=_mm_mul_ps(uo,uc);(ul)=//////
_mm_loadu_ps((float const*)(tu));(um)=_mm_loadu_ps((float const*)(tu+4));(un)=
_mm_loadu_ps((float const*)(tu+(2*4)));(uo)=_mm_loadu_ps((float const*)(tu+(3*4
)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ud));(ui)=_mm_add_ps(ui,_mm_mul_ps(um,ud))
;(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ud));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ud));(
ul)=_mm_loadu_ps((float const*)(tw));(um)=_mm_loadu_ps((float const*)(tw+4));(
un)=_mm_loadu_ps((float const*)(tw+(2*4)));(uo)=_mm_loadu_ps((float const*)(tw+
(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ue));(ui)=_mm_add_ps(ui,_mm_mul_ps(um,
ue));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ue));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ue)
);(ul)=_mm_loadu_ps((float const*)(ty));(um)=_mm_loadu_ps((float const*)(ty+4))
;(un)=_mm_loadu_ps((float const*)(ty+(2*4)));(uo)=_mm_loadu_ps((float const*)(
ty+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,uf));(ui)=_mm_add_ps(ui,_mm_mul_ps(
um,uf));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,uf));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,
uf));(ul)=_mm_loadu_ps((float const*)(ua));(um)=_mm_loadu_ps((float const*)(ua+
4));(un)=_mm_loadu_ps((float const*)(ua+(2*4)));(uo)=_mm_loadu_ps((float const*
)(ua+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ug));(ui)=_mm_add_ps(ui,/////////
_mm_mul_ps(um,ug));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ug));(uk)=_mm_add_ps(uk,///
_mm_mul_ps(uo,ug));_mm_storeu_ps((float*)(tr),uh);_mm_storeu_ps((float*)(tr+4),
ui);_mm_storeu_ps((float*)(tr+(2*4)),uj);_mm_storeu_ps((float*)(tr+(3*4)),uk);
tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);}_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){__m128//
up,uq;__asm__(""::"r"(tr));(uq)=_mm_loadu_ps((float const*)(ts));(up)=/////////
_mm_mul_ps(uq,(uc));(uq)=_mm_loadu_ps((float const*)(tu));(up)=_mm_add_ps(up,//
_mm_mul_ps(uq,(ud)));(uq)=_mm_loadu_ps((float const*)(tw));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(ue)));(uq)=_mm_loadu_ps((float const*)(ty));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(uf)));(uq)=_mm_loadu_ps((float const*)(ua));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(ug)));_mm_storeu_ps((float*)(tr),up);tr+=4;ts+=4;tu+=4;tw+=4;ty
+=4;ua+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float ur
;__asm__(""::"r"(tr));ur=ts[0]*tt;ur+=tu[0]*tv;ur+=tw[0]*tx;ur+=ty[0]*tz;ur+=ua
[0]*ub;tr[0]=ur;++tr;++ts;++tu;++tw;++ty;++ua;}}static void re(float**tn,float
const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to
[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv=tn[2];////
float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3];float*__restrict__ tz
=tn[4];float ua=to[4];{__m128 ub=_mm_set_ps1(ts);__m128 uc=_mm_set_ps1(tu);////
__m128 ud=_mm_set_ps1(tw);__m128 ue=_mm_set_ps1(ty);__m128 uf=_mm_set_ps1(ua);
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*
4)){__m128 ug,uh,ui,uj,uk,ul,um,un;__asm__(""::"r"(tr));(uk)=_mm_loadu_ps((////
float const*)(tp));(ul)=_mm_loadu_ps((float const*)(tp+4));(um)=_mm_loadu_ps((
float const*)(tp+(2*4)));(un)=_mm_loadu_ps((float const*)(tp+(3*4)));(ug)=/////
_mm_loadu_ps((float const*)(tr));(uh)=_mm_loadu_ps((float const*)(tr+4));(ui)=
_mm_loadu_ps((float const*)(tr+(2*4)));(uj)=_mm_loadu_ps((float const*)(tr+(3*4
)));(ug)=_mm_add_ps(ug,_mm_mul_ps(uk,ub));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ub))
;(ui)=_mm_add_ps(ui,_mm_mul_ps(um,ub));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ub));//
_mm_storeu_ps((float*)(tr),ug);_mm_storeu_ps((float*)(tr+4),uh);_mm_storeu_ps((
float*)(tr+(2*4)),ui);_mm_storeu_ps((float*)(tr+(3*4)),uj);(ug)=_mm_loadu_ps((
float const*)(tt));(uh)=_mm_loadu_ps((float const*)(tt+4));(ui)=_mm_loadu_ps((
float const*)(tt+(2*4)));(uj)=_mm_loadu_ps((float const*)(tt+(3*4)));(ug)=/////
_mm_add_ps(ug,_mm_mul_ps(uk,uc));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,uc));(ui)=///
_mm_add_ps(ui,_mm_mul_ps(um,uc));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,uc));////////
_mm_storeu_ps((float*)(tt),ug);_mm_storeu_ps((float*)(tt+4),uh);_mm_storeu_ps((
float*)(tt+(2*4)),ui);_mm_storeu_ps((float*)(tt+(3*4)),uj);(ug)=_mm_loadu_ps((
float const*)(tv));(uh)=_mm_loadu_ps((float const*)(tv+4));(ui)=_mm_loadu_ps((
float const*)(tv+(2*4)));(uj)=_mm_loadu_ps((float const*)(tv+(3*4)));(ug)=/////
_mm_add_ps(ug,_mm_mul_ps(uk,ud));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ud));(ui)=///
_mm_add_ps(ui,_mm_mul_ps(um,ud));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ud));////////
_mm_storeu_ps((float*)(tv),ug);_mm_storeu_ps((float*)(tv+4),uh);_mm_storeu_ps((
float*)(tv+(2*4)),ui);_mm_storeu_ps((float*)(tv+(3*4)),uj);(ug)=_mm_loadu_ps((
float const*)(tx));(uh)=_mm_loadu_ps((float const*)(tx+4));(ui)=_mm_loadu_ps((
float const*)(tx+(2*4)));(uj)=_mm_loadu_ps((float const*)(tx+(3*4)));(ug)=/////
_mm_add_ps(ug,_mm_mul_ps(uk,ue));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ue));(ui)=///
_mm_add_ps(ui,_mm_mul_ps(um,ue));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ue));////////
_mm_storeu_ps((float*)(tx),ug);_mm_storeu_ps((float*)(tx+4),uh);_mm_storeu_ps((
float*)(tx+(2*4)),ui);_mm_storeu_ps((float*)(tx+(3*4)),uj);(ug)=_mm_loadu_ps((
float const*)(tz));(uh)=_mm_loadu_ps((float const*)(tz+4));(ui)=_mm_loadu_ps((
float const*)(tz+(2*4)));(uj)=_mm_loadu_ps((float const*)(tz+(3*4)));(ug)=/////
_mm_add_ps(ug,_mm_mul_ps(uk,uf));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,uf));(ui)=///
_mm_add_ps(ui,_mm_mul_ps(um,uf));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,uf));////////
_mm_storeu_ps((float*)(tz),ug);_mm_storeu_ps((float*)(tz+4),uh);_mm_storeu_ps((
float*)(tz+(2*4)),ui);_mm_storeu_ps((float*)(tz+(3*4)),uj);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);}_Pragma("GCC unroll 1")_Pragma(///////
"GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 uo,up;__asm__(""::"r"(tr
));(up)=_mm_loadu_ps((float const*)(tp));(uo)=_mm_loadu_ps((float const*)(tr));
(uo)=_mm_add_ps(uo,_mm_mul_ps(up,(ub)));_mm_storeu_ps((float*)(tr),uo);(uo)=///
_mm_loadu_ps((float const*)(tt));(uo)=_mm_add_ps(uo,_mm_mul_ps(up,(uc)));//////
_mm_storeu_ps((float*)(tt),uo);(uo)=_mm_loadu_ps((float const*)(tv));(uo)=/////
_mm_add_ps(uo,_mm_mul_ps(up,(ud)));_mm_storeu_ps((float*)(tv),uo);(uo)=////////
_mm_loadu_ps((float const*)(tx));(uo)=_mm_add_ps(uo,_mm_mul_ps(up,(ue)));//////
_mm_storeu_ps((float*)(tx),uo);(uo)=_mm_loadu_ps((float const*)(tz));(uo)=/////
_mm_add_ps(uo,_mm_mul_ps(up,(uf)));_mm_storeu_ps((float*)(tz),uo);tp+=4;tr+=4;
tt+=4;tv+=4;tx+=4;tz+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(
tp<tq){float uq=tp[0];__asm__(""::"r"(tr));tr[0]+=(uq*ts);tt[0]+=(uq*tu);tv[0]
+=(uq*tw);tx[0]+=(uq*ty);tz[0]+=(uq*ua);++tp;++tr;++tt;++tv;++tx;++tz;}}static
void rf(float*tn,float const*to,float const**tp,float const*tq){float*/////////
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];float const*tw=tp[2];float tx=to[2];float const*ty=tp[3];float//
tz=to[3];float const*ua=tp[4];float ub=to[4];{__m128 uc=_mm_set_ps1(tt);__m128
ud=_mm_set_ps1(tv);__m128 ue=_mm_set_ps1(tx);__m128 uf=_mm_set_ps1(tz);__m128//
ug=_mm_set_ps1(ub);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)
tq-(char*)ts)>=(16*4)){__m128 uh,ui,uj,uk,ul,um,un,uo;__asm__(""::"r"(tr));////
_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(
ty+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);(uh)=/////
_mm_loadu_ps((float const*)(tr));(ui)=_mm_loadu_ps((float const*)(tr+4));(uj)=
_mm_loadu_ps((float const*)(tr+(2*4)));(uk)=_mm_loadu_ps((float const*)(tr+(3*4
)));(ul)=_mm_loadu_ps((float const*)(ts));(um)=_mm_loadu_ps((float const*)(ts+4
));(un)=_mm_loadu_ps((float const*)(ts+(2*4)));(uo)=_mm_loadu_ps((float const*)
(ts+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,uc));(ui)=_mm_add_ps(ui,_mm_mul_ps
(um,uc));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,uc));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo
,uc));(ul)=_mm_loadu_ps((float const*)(tu));(um)=_mm_loadu_ps((float const*)(tu
+4));(un)=_mm_loadu_ps((float const*)(tu+(2*4)));(uo)=_mm_loadu_ps((float const
*)(tu+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ud));(ui)=_mm_add_ps(ui,////////
_mm_mul_ps(um,ud));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ud));(uk)=_mm_add_ps(uk,///
_mm_mul_ps(uo,ud));(ul)=_mm_loadu_ps((float const*)(tw));(um)=_mm_loadu_ps((///
float const*)(tw+4));(un)=_mm_loadu_ps((float const*)(tw+(2*4)));(uo)=/////////
_mm_loadu_ps((float const*)(tw+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,ue));(
ui)=_mm_add_ps(ui,_mm_mul_ps(um,ue));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ue));(uk)
=_mm_add_ps(uk,_mm_mul_ps(uo,ue));(ul)=_mm_loadu_ps((float const*)(ty));(um)=//
_mm_loadu_ps((float const*)(ty+4));(un)=_mm_loadu_ps((float const*)(ty+(2*4)));
(uo)=_mm_loadu_ps((float const*)(ty+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,uf
));(ui)=_mm_add_ps(ui,_mm_mul_ps(um,uf));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,uf));
(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uf));(ul)=_mm_loadu_ps((float const*)(ua));(um
)=_mm_loadu_ps((float const*)(ua+4));(un)=_mm_loadu_ps((float const*)(ua+(2*4))
);(uo)=_mm_loadu_ps((float const*)(ua+(3*4)));(uh)=_mm_add_ps(uh,_mm_mul_ps(ul,
ug));(ui)=_mm_add_ps(ui,_mm_mul_ps(um,ug));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ug)
);(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ug));_mm_storeu_ps((float*)(tr),uh);////////
_mm_storeu_ps((float*)(tr+4),ui);_mm_storeu_ps((float*)(tr+(2*4)),uj);/////////
_mm_storeu_ps((float*)(tr+(3*4)),uk);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty
+=(4*4);ua+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)
tq-(char*)ts)>=16){__m128 up,uq;__asm__(""::"r"(tr));(up)=_mm_loadu_ps((float//
const*)(tr));(uq)=_mm_loadu_ps((float const*)(ts));(up)=_mm_add_ps(up,/////////
_mm_mul_ps(uq,(uc)));(uq)=_mm_loadu_ps((float const*)(tu));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(ud)));(uq)=_mm_loadu_ps((float const*)(tw));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(ue)));(uq)=_mm_loadu_ps((float const*)(ty));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(uf)));(uq)=_mm_loadu_ps((float const*)(ua));(up)=_mm_add_ps(up,
_mm_mul_ps(uq,(ug)));_mm_storeu_ps((float*)(tr),up);tr+=4;ts+=4;tu+=4;tw+=4;ty
+=4;ua+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float ur
;__asm__(""::"r"(tr));ur=tr[0]+ts[0]*tt;ur+=tu[0]*tv;ur+=tw[0]*tx;ur+=ty[0]*tz;
ur+=ua[0]*ub;tr[0]=ur;++tr;++ts;++tu;++tw;++ty;++ua;}}static void rg(float**tn,
float const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];float
ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv=tn[2]
;float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3];float*__restrict__//
tz=tn[4];float ua=to[4];float*__restrict__ ub=tn[5];float uc=to[5];{__m128 ud=
_mm_set_ps1(ts);__m128 ue=_mm_set_ps1(tu);__m128 uf=_mm_set_ps1(tw);__m128 ug=
_mm_set_ps1(ty);__m128 uh=_mm_set_ps1(ua);__m128 ui=_mm_set_ps1(uc);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){////
__m128 uj,uk,ul,um,un,uo,up,uq;__asm__(""::"r"(tr));(un)=_mm_loadu_ps((float///
const*)(tp));(uo)=_mm_loadu_ps((float const*)(tp+4));(up)=_mm_loadu_ps((float//
const*)(tp+(2*4)));(uq)=_mm_loadu_ps((float const*)(tp+(3*4)));(uj)=_mm_mul_ps(
un,ud);(uk)=_mm_mul_ps(uo,ud);(ul)=_mm_mul_ps(up,ud);(um)=_mm_mul_ps(uq,ud);///
_mm_storeu_ps((float*)(tr),uj);_mm_storeu_ps((float*)(tr+4),uk);_mm_storeu_ps((
float*)(tr+(2*4)),ul);_mm_storeu_ps((float*)(tr+(3*4)),um);(uj)=_mm_mul_ps(un,
ue);(uk)=_mm_mul_ps(uo,ue);(ul)=_mm_mul_ps(up,ue);(um)=_mm_mul_ps(uq,ue);//////
_mm_storeu_ps((float*)(tt),uj);_mm_storeu_ps((float*)(tt+4),uk);_mm_storeu_ps((
float*)(tt+(2*4)),ul);_mm_storeu_ps((float*)(tt+(3*4)),um);(uj)=_mm_mul_ps(un,
uf);(uk)=_mm_mul_ps(uo,uf);(ul)=_mm_mul_ps(up,uf);(um)=_mm_mul_ps(uq,uf);//////
_mm_storeu_ps((float*)(tv),uj);_mm_storeu_ps((float*)(tv+4),uk);_mm_storeu_ps((
float*)(tv+(2*4)),ul);_mm_storeu_ps((float*)(tv+(3*4)),um);(uj)=_mm_mul_ps(un,
ug);(uk)=_mm_mul_ps(uo,ug);(ul)=_mm_mul_ps(up,ug);(um)=_mm_mul_ps(uq,ug);//////
_mm_storeu_ps((float*)(tx),uj);_mm_storeu_ps((float*)(tx+4),uk);_mm_storeu_ps((
float*)(tx+(2*4)),ul);_mm_storeu_ps((float*)(tx+(3*4)),um);(uj)=_mm_mul_ps(un,
uh);(uk)=_mm_mul_ps(uo,uh);(ul)=_mm_mul_ps(up,uh);(um)=_mm_mul_ps(uq,uh);//////
_mm_storeu_ps((float*)(tz),uj);_mm_storeu_ps((float*)(tz+4),uk);_mm_storeu_ps((
float*)(tz+(2*4)),ul);_mm_storeu_ps((float*)(tz+(3*4)),um);(uj)=_mm_mul_ps(un,
ui);(uk)=_mm_mul_ps(uo,ui);(ul)=_mm_mul_ps(up,ui);(um)=_mm_mul_ps(uq,ui);//////
_mm_storeu_ps((float*)(ub),uj);_mm_storeu_ps((float*)(ub+4),uk);_mm_storeu_ps((
float*)(ub+(2*4)),ul);_mm_storeu_ps((float*)(ub+(3*4)),um);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);}_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 ur,us;__asm__(""
::"r"(tr));(us)=_mm_loadu_ps((float const*)(tp));(ur)=_mm_mul_ps(us,(ud));/////
_mm_storeu_ps((float*)(tr),ur);(ur)=_mm_mul_ps(us,(ue));_mm_storeu_ps((float*)(
tt),ur);(ur)=_mm_mul_ps(us,(uf));_mm_storeu_ps((float*)(tv),ur);(ur)=_mm_mul_ps
(us,(ug));_mm_storeu_ps((float*)(tx),ur);(ur)=_mm_mul_ps(us,(uh));_mm_storeu_ps
((float*)(tz),ur);(ur)=_mm_mul_ps(us,(ui));_mm_storeu_ps((float*)(ub),ur);tp+=4
;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;}}_Pragma("GCC unroll 1")_Pragma(/////////
"GCC novector")while(tp<tq){float ut=tp[0];__asm__(""::"r"(tr));tr[0]=(ut*ts);
tt[0]=(ut*tu);tv[0]=(ut*tw);tx[0]=(ut*ty);tz[0]=(ut*ua);ub[0]=(ut*uc);++tp;++tr
;++tt;++tv;++tx;++tz;++ub;}}static void rh(float*tn,float const*to,float const*
*tp,float const*tq){float*__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0
];float const*tu=tp[1];float tv=to[1];float const*tw=tp[2];float tx=to[2];float
const*ty=tp[3];float tz=to[3];float const*ua=tp[4];float ub=to[4];float const*
uc=tp[5];float ud=to[5];{__m128 ue=_mm_set_ps1(tt);__m128 uf=_mm_set_ps1(tv);//
__m128 ug=_mm_set_ps1(tx);__m128 uh=_mm_set_ps1(tz);__m128 ui=_mm_set_ps1(ub);
__m128 uj=_mm_set_ps1(ud);_Pragma("GCC unroll 1")_Pragma("GCC novector")while((
(char*)tq-(char*)ts)>=(16*4)){__m128 uk,ul,um,un,uo,up,uq,ur;__asm__(""::"r"(tr
));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4))
,_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)
(ty+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);/////////
_mm_prefetch((char*)(uc+(16*4)),_MM_HINT_T0);(uo)=_mm_loadu_ps((float const*)(
ts));(up)=_mm_loadu_ps((float const*)(ts+4));(uq)=_mm_loadu_ps((float const*)(
ts+(2*4)));(ur)=_mm_loadu_ps((float const*)(ts+(3*4)));(uk)=_mm_mul_ps(uo,ue);(
ul)=_mm_mul_ps(up,ue);(um)=_mm_mul_ps(uq,ue);(un)=_mm_mul_ps(ur,ue);(uo)=//////
_mm_loadu_ps((float const*)(tu));(up)=_mm_loadu_ps((float const*)(tu+4));(uq)=
_mm_loadu_ps((float const*)(tu+(2*4)));(ur)=_mm_loadu_ps((float const*)(tu+(3*4
)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uf));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,uf))
;(um)=_mm_add_ps(um,_mm_mul_ps(uq,uf));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uf));(
uo)=_mm_loadu_ps((float const*)(tw));(up)=_mm_loadu_ps((float const*)(tw+4));(
uq)=_mm_loadu_ps((float const*)(tw+(2*4)));(ur)=_mm_loadu_ps((float const*)(tw+
(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ug));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,
ug));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ug));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ug)
);(uo)=_mm_loadu_ps((float const*)(ty));(up)=_mm_loadu_ps((float const*)(ty+4))
;(uq)=_mm_loadu_ps((float const*)(ty+(2*4)));(ur)=_mm_loadu_ps((float const*)(
ty+(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uh));(ul)=_mm_add_ps(ul,_mm_mul_ps(
up,uh));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uh));(un)=_mm_add_ps(un,_mm_mul_ps(ur,
uh));(uo)=_mm_loadu_ps((float const*)(ua));(up)=_mm_loadu_ps((float const*)(ua+
4));(uq)=_mm_loadu_ps((float const*)(ua+(2*4)));(ur)=_mm_loadu_ps((float const*
)(ua+(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ui));(ul)=_mm_add_ps(ul,/////////
_mm_mul_ps(up,ui));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ui));(un)=_mm_add_ps(un,///
_mm_mul_ps(ur,ui));(uo)=_mm_loadu_ps((float const*)(uc));(up)=_mm_loadu_ps((///
float const*)(uc+4));(uq)=_mm_loadu_ps((float const*)(uc+(2*4)));(ur)=/////////
_mm_loadu_ps((float const*)(uc+(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uj));(
ul)=_mm_add_ps(ul,_mm_mul_ps(up,uj));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uj));(un)
=_mm_add_ps(un,_mm_mul_ps(ur,uj));_mm_storeu_ps((float*)(tr),uk);_mm_storeu_ps(
(float*)(tr+4),ul);_mm_storeu_ps((float*)(tr+(2*4)),um);_mm_storeu_ps((float*)(
tr+(3*4)),un);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);uc+=(
4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)
>=16){__m128 us,ut;__asm__(""::"r"(tr));(ut)=_mm_loadu_ps((float const*)(ts));(
us)=_mm_mul_ps(ut,(ue));(ut)=_mm_loadu_ps((float const*)(tu));(us)=_mm_add_ps(
us,_mm_mul_ps(ut,(uf)));(ut)=_mm_loadu_ps((float const*)(tw));(us)=_mm_add_ps(
us,_mm_mul_ps(ut,(ug)));(ut)=_mm_loadu_ps((float const*)(ty));(us)=_mm_add_ps(
us,_mm_mul_ps(ut,(uh)));(ut)=_mm_loadu_ps((float const*)(ua));(us)=_mm_add_ps(
us,_mm_mul_ps(ut,(ui)));(ut)=_mm_loadu_ps((float const*)(uc));(us)=_mm_add_ps(
us,_mm_mul_ps(ut,(uj)));_mm_storeu_ps((float*)(tr),us);tr+=4;ts+=4;tu+=4;tw+=4;
ty+=4;ua+=4;uc+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){
float uu;__asm__(""::"r"(tr));uu=ts[0]*tt;uu+=tu[0]*tv;uu+=tw[0]*tx;uu+=ty[0]*
tz;uu+=ua[0]*ub;uu+=uc[0]*ud;tr[0]=uu;++tr;++ts;++tu;++tw;++ty;++ua;++uc;}}////
static void rj(float**tn,float const*to,float const*tp,float const*tq){float*//
__restrict__ tr=tn[0];float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1]
;float*__restrict__ tv=tn[2];float tw=to[2];float*__restrict__ tx=tn[3];float//
ty=to[3];float*__restrict__ tz=tn[4];float ua=to[4];float*__restrict__ ub=tn[5]
;float uc=to[5];{__m128 ud=_mm_set_ps1(ts);__m128 ue=_mm_set_ps1(tu);__m128 uf=
_mm_set_ps1(tw);__m128 ug=_mm_set_ps1(ty);__m128 uh=_mm_set_ps1(ua);__m128 ui=
_mm_set_ps1(uc);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-
(char*)tp)>=(16*4)){__m128 uj,uk,ul,um,un,uo,up,uq;__asm__(""::"r"(tr));(un)=//
_mm_loadu_ps((float const*)(tp));(uo)=_mm_loadu_ps((float const*)(tp+4));(up)=
_mm_loadu_ps((float const*)(tp+(2*4)));(uq)=_mm_loadu_ps((float const*)(tp+(3*4
)));(uj)=_mm_loadu_ps((float const*)(tr));(uk)=_mm_loadu_ps((float const*)(tr+4
));(ul)=_mm_loadu_ps((float const*)(tr+(2*4)));(um)=_mm_loadu_ps((float const*)
(tr+(3*4)));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ud));(uk)=_mm_add_ps(uk,_mm_mul_ps
(uo,ud));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,ud));(um)=_mm_add_ps(um,_mm_mul_ps(uq
,ud));_mm_storeu_ps((float*)(tr),uj);_mm_storeu_ps((float*)(tr+4),uk);/////////
_mm_storeu_ps((float*)(tr+(2*4)),ul);_mm_storeu_ps((float*)(tr+(3*4)),um);(uj)=
_mm_loadu_ps((float const*)(tt));(uk)=_mm_loadu_ps((float const*)(tt+4));(ul)=
_mm_loadu_ps((float const*)(tt+(2*4)));(um)=_mm_loadu_ps((float const*)(tt+(3*4
)));(uj)=_mm_add_ps(uj,_mm_mul_ps(un,ue));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ue))
;(ul)=_mm_add_ps(ul,_mm_mul_ps(up,ue));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ue));//
_mm_storeu_ps((float*)(tt),uj);_mm_storeu_ps((float*)(tt+4),uk);_mm_storeu_ps((
float*)(tt+(2*4)),ul);_mm_storeu_ps((float*)(tt+(3*4)),um);(uj)=_mm_loadu_ps((
float const*)(tv));(uk)=_mm_loadu_ps((float const*)(tv+4));(ul)=_mm_loadu_ps((
float const*)(tv+(2*4)));(um)=_mm_loadu_ps((float const*)(tv+(3*4)));(uj)=/////
_mm_add_ps(uj,_mm_mul_ps(un,uf));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uf));(ul)=///
_mm_add_ps(ul,_mm_mul_ps(up,uf));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uf));////////
_mm_storeu_ps((float*)(tv),uj);_mm_storeu_ps((float*)(tv+4),uk);_mm_storeu_ps((
float*)(tv+(2*4)),ul);_mm_storeu_ps((float*)(tv+(3*4)),um);(uj)=_mm_loadu_ps((
float const*)(tx));(uk)=_mm_loadu_ps((float const*)(tx+4));(ul)=_mm_loadu_ps((
float const*)(tx+(2*4)));(um)=_mm_loadu_ps((float const*)(tx+(3*4)));(uj)=/////
_mm_add_ps(uj,_mm_mul_ps(un,ug));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ug));(ul)=///
_mm_add_ps(ul,_mm_mul_ps(up,ug));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ug));////////
_mm_storeu_ps((float*)(tx),uj);_mm_storeu_ps((float*)(tx+4),uk);_mm_storeu_ps((
float*)(tx+(2*4)),ul);_mm_storeu_ps((float*)(tx+(3*4)),um);(uj)=_mm_loadu_ps((
float const*)(tz));(uk)=_mm_loadu_ps((float const*)(tz+4));(ul)=_mm_loadu_ps((
float const*)(tz+(2*4)));(um)=_mm_loadu_ps((float const*)(tz+(3*4)));(uj)=/////
_mm_add_ps(uj,_mm_mul_ps(un,uh));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uh));(ul)=///
_mm_add_ps(ul,_mm_mul_ps(up,uh));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uh));////////
_mm_storeu_ps((float*)(tz),uj);_mm_storeu_ps((float*)(tz+4),uk);_mm_storeu_ps((
float*)(tz+(2*4)),ul);_mm_storeu_ps((float*)(tz+(3*4)),um);(uj)=_mm_loadu_ps((
float const*)(ub));(uk)=_mm_loadu_ps((float const*)(ub+4));(ul)=_mm_loadu_ps((
float const*)(ub+(2*4)));(um)=_mm_loadu_ps((float const*)(ub+(3*4)));(uj)=/////
_mm_add_ps(uj,_mm_mul_ps(un,ui));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ui));(ul)=///
_mm_add_ps(ul,_mm_mul_ps(up,ui));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ui));////////
_mm_storeu_ps((float*)(ub),uj);_mm_storeu_ps((float*)(ub+4),uk);_mm_storeu_ps((
float*)(ub+(2*4)),ul);_mm_storeu_ps((float*)(ub+(3*4)),um);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);}_Pragma("GCC unroll 1")/////
_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 ur,us;__asm__(""
::"r"(tr));(us)=_mm_loadu_ps((float const*)(tp));(ur)=_mm_loadu_ps((float const
*)(tr));(ur)=_mm_add_ps(ur,_mm_mul_ps(us,(ud)));_mm_storeu_ps((float*)(tr),ur);
(ur)=_mm_loadu_ps((float const*)(tt));(ur)=_mm_add_ps(ur,_mm_mul_ps(us,(ue)));
_mm_storeu_ps((float*)(tt),ur);(ur)=_mm_loadu_ps((float const*)(tv));(ur)=/////
_mm_add_ps(ur,_mm_mul_ps(us,(uf)));_mm_storeu_ps((float*)(tv),ur);(ur)=////////
_mm_loadu_ps((float const*)(tx));(ur)=_mm_add_ps(ur,_mm_mul_ps(us,(ug)));//////
_mm_storeu_ps((float*)(tx),ur);(ur)=_mm_loadu_ps((float const*)(tz));(ur)=/////
_mm_add_ps(ur,_mm_mul_ps(us,(uh)));_mm_storeu_ps((float*)(tz),ur);(ur)=////////
_mm_loadu_ps((float const*)(ub));(ur)=_mm_add_ps(ur,_mm_mul_ps(us,(ui)));//////
_mm_storeu_ps((float*)(ub),ur);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;}}////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float ut=tp[0];/////
__asm__(""::"r"(tr));tr[0]+=(ut*ts);tt[0]+=(ut*tu);tv[0]+=(ut*tw);tx[0]+=(ut*ty
);tz[0]+=(ut*ua);ub[0]+=(ut*uc);++tp;++tr;++tt;++tv;++tx;++tz;++ub;}}static////
void rk(float*tn,float const*to,float const**tp,float const*tq){float*/////////
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];float const*tw=tp[2];float tx=to[2];float const*ty=tp[3];float//
tz=to[3];float const*ua=tp[4];float ub=to[4];float const*uc=tp[5];float ud=to[5
];{__m128 ue=_mm_set_ps1(tt);__m128 uf=_mm_set_ps1(tv);__m128 ug=_mm_set_ps1(tx
);__m128 uh=_mm_set_ps1(tz);__m128 ui=_mm_set_ps1(ub);__m128 uj=_mm_set_ps1(ud)
;_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16
*4)){__m128 uk,ul,um,un,uo,up,uq,ur;__asm__(""::"r"(tr));_mm_prefetch((char*)(
ts+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);//////////
_mm_prefetch((char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(
uc+(16*4)),_MM_HINT_T0);(uk)=_mm_loadu_ps((float const*)(tr));(ul)=_mm_loadu_ps
((float const*)(tr+4));(um)=_mm_loadu_ps((float const*)(tr+(2*4)));(un)=///////
_mm_loadu_ps((float const*)(tr+(3*4)));(uo)=_mm_loadu_ps((float const*)(ts));(
up)=_mm_loadu_ps((float const*)(ts+4));(uq)=_mm_loadu_ps((float const*)(ts+(2*4
)));(ur)=_mm_loadu_ps((float const*)(ts+(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(
uo,ue));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,ue));(um)=_mm_add_ps(um,_mm_mul_ps(uq,
ue));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ue));(uo)=_mm_loadu_ps((float const*)(tu)
);(up)=_mm_loadu_ps((float const*)(tu+4));(uq)=_mm_loadu_ps((float const*)(tu+(
2*4)));(ur)=_mm_loadu_ps((float const*)(tu+(3*4)));(uk)=_mm_add_ps(uk,/////////
_mm_mul_ps(uo,uf));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,uf));(um)=_mm_add_ps(um,///
_mm_mul_ps(uq,uf));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uf));(uo)=_mm_loadu_ps((///
float const*)(tw));(up)=_mm_loadu_ps((float const*)(tw+4));(uq)=_mm_loadu_ps((
float const*)(tw+(2*4)));(ur)=_mm_loadu_ps((float const*)(tw+(3*4)));(uk)=/////
_mm_add_ps(uk,_mm_mul_ps(uo,ug));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,ug));(um)=///
_mm_add_ps(um,_mm_mul_ps(uq,ug));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ug));(uo)=///
_mm_loadu_ps((float const*)(ty));(up)=_mm_loadu_ps((float const*)(ty+4));(uq)=
_mm_loadu_ps((float const*)(ty+(2*4)));(ur)=_mm_loadu_ps((float const*)(ty+(3*4
)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uh));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,uh))
;(um)=_mm_add_ps(um,_mm_mul_ps(uq,uh));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uh));(
uo)=_mm_loadu_ps((float const*)(ua));(up)=_mm_loadu_ps((float const*)(ua+4));(
uq)=_mm_loadu_ps((float const*)(ua+(2*4)));(ur)=_mm_loadu_ps((float const*)(ua+
(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,ui));(ul)=_mm_add_ps(ul,_mm_mul_ps(up,
ui));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ui));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ui)
);(uo)=_mm_loadu_ps((float const*)(uc));(up)=_mm_loadu_ps((float const*)(uc+4))
;(uq)=_mm_loadu_ps((float const*)(uc+(2*4)));(ur)=_mm_loadu_ps((float const*)(
uc+(3*4)));(uk)=_mm_add_ps(uk,_mm_mul_ps(uo,uj));(ul)=_mm_add_ps(ul,_mm_mul_ps(
up,uj));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uj));(un)=_mm_add_ps(un,_mm_mul_ps(ur,
uj));_mm_storeu_ps((float*)(tr),uk);_mm_storeu_ps((float*)(tr+4),ul);//////////
_mm_storeu_ps((float*)(tr+(2*4)),um);_mm_storeu_ps((float*)(tr+(3*4)),un);tr+=(
4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);uc+=(4*4);}_Pragma(/////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){__m128//
us,ut;__asm__(""::"r"(tr));(us)=_mm_loadu_ps((float const*)(tr));(ut)=/////////
_mm_loadu_ps((float const*)(ts));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(ue)));(ut)=
_mm_loadu_ps((float const*)(tu));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(uf)));(ut)=
_mm_loadu_ps((float const*)(tw));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(ug)));(ut)=
_mm_loadu_ps((float const*)(ty));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(uh)));(ut)=
_mm_loadu_ps((float const*)(ua));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(ui)));(ut)=
_mm_loadu_ps((float const*)(uc));(us)=_mm_add_ps(us,_mm_mul_ps(ut,(uj)));//////
_mm_storeu_ps((float*)(tr),us);tr+=4;ts+=4;tu+=4;tw+=4;ty+=4;ua+=4;uc+=4;}}////
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float uu;__asm__(""
::"r"(tr));uu=tr[0]+ts[0]*tt;uu+=tu[0]*tv;uu+=tw[0]*tx;uu+=ty[0]*tz;uu+=ua[0]*
ub;uu+=uc[0]*ud;tr[0]=uu;++tr;++ts;++tu;++tw;++ty;++ua;++uc;}}static void rl(//
float**tn,float const*to,float const*tp,float const*tq){float*__restrict__ tr=
tn[0];float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*/////////
__restrict__ tv=tn[2];float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3]
;float*__restrict__ tz=tn[4];float ua=to[4];float*__restrict__ ub=tn[5];float//
uc=to[5];float*__restrict__ ud=tn[6];float ue=to[6];{__m128 uf=_mm_set_ps1(ts);
__m128 ug=_mm_set_ps1(tu);__m128 uh=_mm_set_ps1(tw);__m128 ui=_mm_set_ps1(ty);
__m128 uj=_mm_set_ps1(ua);__m128 uk=_mm_set_ps1(uc);__m128 ul=_mm_set_ps1(ue);
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*
4)){__m128 um,un,uo,up,uq,ur,us,ut;__asm__(""::"r"(tr));(uq)=_mm_loadu_ps((////
float const*)(tp));(ur)=_mm_loadu_ps((float const*)(tp+4));(us)=_mm_loadu_ps((
float const*)(tp+(2*4)));(ut)=_mm_loadu_ps((float const*)(tp+(3*4)));(um)=/////
_mm_mul_ps(uq,uf);(un)=_mm_mul_ps(ur,uf);(uo)=_mm_mul_ps(us,uf);(up)=_mm_mul_ps
(ut,uf);_mm_storeu_ps((float*)(tr),um);_mm_storeu_ps((float*)(tr+4),un);///////
_mm_storeu_ps((float*)(tr+(2*4)),uo);_mm_storeu_ps((float*)(tr+(3*4)),up);(um)=
_mm_mul_ps(uq,ug);(un)=_mm_mul_ps(ur,ug);(uo)=_mm_mul_ps(us,ug);(up)=_mm_mul_ps
(ut,ug);_mm_storeu_ps((float*)(tt),um);_mm_storeu_ps((float*)(tt+4),un);///////
_mm_storeu_ps((float*)(tt+(2*4)),uo);_mm_storeu_ps((float*)(tt+(3*4)),up);(um)=
_mm_mul_ps(uq,uh);(un)=_mm_mul_ps(ur,uh);(uo)=_mm_mul_ps(us,uh);(up)=_mm_mul_ps
(ut,uh);_mm_storeu_ps((float*)(tv),um);_mm_storeu_ps((float*)(tv+4),un);///////
_mm_storeu_ps((float*)(tv+(2*4)),uo);_mm_storeu_ps((float*)(tv+(3*4)),up);(um)=
_mm_mul_ps(uq,ui);(un)=_mm_mul_ps(ur,ui);(uo)=_mm_mul_ps(us,ui);(up)=_mm_mul_ps
(ut,ui);_mm_storeu_ps((float*)(tx),um);_mm_storeu_ps((float*)(tx+4),un);///////
_mm_storeu_ps((float*)(tx+(2*4)),uo);_mm_storeu_ps((float*)(tx+(3*4)),up);(um)=
_mm_mul_ps(uq,uj);(un)=_mm_mul_ps(ur,uj);(uo)=_mm_mul_ps(us,uj);(up)=_mm_mul_ps
(ut,uj);_mm_storeu_ps((float*)(tz),um);_mm_storeu_ps((float*)(tz+4),un);///////
_mm_storeu_ps((float*)(tz+(2*4)),uo);_mm_storeu_ps((float*)(tz+(3*4)),up);(um)=
_mm_mul_ps(uq,uk);(un)=_mm_mul_ps(ur,uk);(uo)=_mm_mul_ps(us,uk);(up)=_mm_mul_ps
(ut,uk);_mm_storeu_ps((float*)(ub),um);_mm_storeu_ps((float*)(ub+4),un);///////
_mm_storeu_ps((float*)(ub+(2*4)),uo);_mm_storeu_ps((float*)(ub+(3*4)),up);(um)=
_mm_mul_ps(uq,ul);(un)=_mm_mul_ps(ur,ul);(uo)=_mm_mul_ps(us,ul);(up)=_mm_mul_ps
(ut,ul);_mm_storeu_ps((float*)(ud),um);_mm_storeu_ps((float*)(ud+4),un);///////
_mm_storeu_ps((float*)(ud+(2*4)),uo);_mm_storeu_ps((float*)(ud+(3*4)),up);tp+=(
4*4);tr+=(4*4);tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);ud+=(4*4);}///
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){
__m128 uu,uv;__asm__(""::"r"(tr));(uv)=_mm_loadu_ps((float const*)(tp));(uu)=//
_mm_mul_ps(uv,(uf));_mm_storeu_ps((float*)(tr),uu);(uu)=_mm_mul_ps(uv,(ug));///
_mm_storeu_ps((float*)(tt),uu);(uu)=_mm_mul_ps(uv,(uh));_mm_storeu_ps((float*)(
tv),uu);(uu)=_mm_mul_ps(uv,(ui));_mm_storeu_ps((float*)(tx),uu);(uu)=_mm_mul_ps
(uv,(uj));_mm_storeu_ps((float*)(tz),uu);(uu)=_mm_mul_ps(uv,(uk));_mm_storeu_ps
((float*)(ub),uu);(uu)=_mm_mul_ps(uv,(ul));_mm_storeu_ps((float*)(ud),uu);tp+=4
;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;ud+=4;}}_Pragma("GCC unroll 1")_Pragma(///
"GCC novector")while(tp<tq){float uw=tp[0];__asm__(""::"r"(tr));tr[0]=(uw*ts);
tt[0]=(uw*tu);tv[0]=(uw*tw);tx[0]=(uw*ty);tz[0]=(uw*ua);ub[0]=(uw*uc);ud[0]=(uw
*ue);++tp;++tr;++tt;++tv;++tx;++tz;++ub;++ud;}}static void rm(float*tn,float///
const*to,float const**tp,float const*tq){float*__restrict__ tr=tn;float const*
ts=tp[0];float tt=to[0];float const*tu=tp[1];float tv=to[1];float const*tw=tp[2
];float tx=to[2];float const*ty=tp[3];float tz=to[3];float const*ua=tp[4];float
ub=to[4];float const*uc=tp[5];float ud=to[5];float const*ue=tp[6];float uf=to[6
];{__m128 ug=_mm_set_ps1(tt);__m128 uh=_mm_set_ps1(tv);__m128 ui=_mm_set_ps1(tx
);__m128 uj=_mm_set_ps1(tz);__m128 uk=_mm_set_ps1(ub);__m128 ul=_mm_set_ps1(ud)
;__m128 um=_mm_set_ps1(uf);_Pragma("GCC unroll 1")_Pragma("GCC novector")while(
((char*)tq-(char*)ts)>=(16*4)){__m128 un,uo,up,uq,ur,us,ut,uu;__asm__(""::"r"(
tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4
)),_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char
*)(ty+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);///////
_mm_prefetch((char*)(uc+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ue+(16*4)),//
_MM_HINT_T0);(ur)=_mm_loadu_ps((float const*)(ts));(us)=_mm_loadu_ps((float////
const*)(ts+4));(ut)=_mm_loadu_ps((float const*)(ts+(2*4)));(uu)=_mm_loadu_ps((
float const*)(ts+(3*4)));(un)=_mm_mul_ps(ur,ug);(uo)=_mm_mul_ps(us,ug);(up)=///
_mm_mul_ps(ut,ug);(uq)=_mm_mul_ps(uu,ug);(ur)=_mm_loadu_ps((float const*)(tu));
(us)=_mm_loadu_ps((float const*)(tu+4));(ut)=_mm_loadu_ps((float const*)(tu+(2*
4)));(uu)=_mm_loadu_ps((float const*)(tu+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(
ur,uh));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,uh));(up)=_mm_add_ps(up,_mm_mul_ps(ut,
uh));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uh));(ur)=_mm_loadu_ps((float const*)(tw)
);(us)=_mm_loadu_ps((float const*)(tw+4));(ut)=_mm_loadu_ps((float const*)(tw+(
2*4)));(uu)=_mm_loadu_ps((float const*)(tw+(3*4)));(un)=_mm_add_ps(un,/////////
_mm_mul_ps(ur,ui));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,ui));(up)=_mm_add_ps(up,///
_mm_mul_ps(ut,ui));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ui));(ur)=_mm_loadu_ps((///
float const*)(ty));(us)=_mm_loadu_ps((float const*)(ty+4));(ut)=_mm_loadu_ps((
float const*)(ty+(2*4)));(uu)=_mm_loadu_ps((float const*)(ty+(3*4)));(un)=/////
_mm_add_ps(un,_mm_mul_ps(ur,uj));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,uj));(up)=///
_mm_add_ps(up,_mm_mul_ps(ut,uj));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uj));(ur)=///
_mm_loadu_ps((float const*)(ua));(us)=_mm_loadu_ps((float const*)(ua+4));(ut)=
_mm_loadu_ps((float const*)(ua+(2*4)));(uu)=_mm_loadu_ps((float const*)(ua+(3*4
)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uk));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,uk))
;(up)=_mm_add_ps(up,_mm_mul_ps(ut,uk));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uk));(
ur)=_mm_loadu_ps((float const*)(uc));(us)=_mm_loadu_ps((float const*)(uc+4));(
ut)=_mm_loadu_ps((float const*)(uc+(2*4)));(uu)=_mm_loadu_ps((float const*)(uc+
(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ul));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,
ul));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ul));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ul)
);(ur)=_mm_loadu_ps((float const*)(ue));(us)=_mm_loadu_ps((float const*)(ue+4))
;(ut)=_mm_loadu_ps((float const*)(ue+(2*4)));(uu)=_mm_loadu_ps((float const*)(
ue+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,um));(uo)=_mm_add_ps(uo,_mm_mul_ps(
us,um));(up)=_mm_add_ps(up,_mm_mul_ps(ut,um));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,
um));_mm_storeu_ps((float*)(tr),un);_mm_storeu_ps((float*)(tr+4),uo);//////////
_mm_storeu_ps((float*)(tr+(2*4)),up);_mm_storeu_ps((float*)(tr+(3*4)),uq);tr+=(
4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);uc+=(4*4);ue+=(4*4);}///
_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=16){
__m128 uv,uw;__asm__(""::"r"(tr));(uw)=_mm_loadu_ps((float const*)(ts));(uv)=//
_mm_mul_ps(uw,(ug));(uw)=_mm_loadu_ps((float const*)(tu));(uv)=_mm_add_ps(uv,//
_mm_mul_ps(uw,(uh)));(uw)=_mm_loadu_ps((float const*)(tw));(uv)=_mm_add_ps(uv,
_mm_mul_ps(uw,(ui)));(uw)=_mm_loadu_ps((float const*)(ty));(uv)=_mm_add_ps(uv,
_mm_mul_ps(uw,(uj)));(uw)=_mm_loadu_ps((float const*)(ua));(uv)=_mm_add_ps(uv,
_mm_mul_ps(uw,(uk)));(uw)=_mm_loadu_ps((float const*)(uc));(uv)=_mm_add_ps(uv,
_mm_mul_ps(uw,(ul)));(uw)=_mm_loadu_ps((float const*)(ue));(uv)=_mm_add_ps(uv,
_mm_mul_ps(uw,(um)));_mm_storeu_ps((float*)(tr),uv);tr+=4;ts+=4;tu+=4;tw+=4;ty
+=4;ua+=4;uc+=4;ue+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<
tq){float ux;__asm__(""::"r"(tr));ux=ts[0]*tt;ux+=tu[0]*tv;ux+=tw[0]*tx;ux+=ty[
0]*tz;ux+=ua[0]*ub;ux+=uc[0]*ud;ux+=ue[0]*uf;tr[0]=ux;++tr;++ts;++tu;++tw;++ty;
++ua;++uc;++ue;}}static void rn(float**tn,float const*to,float const*tp,float//
const*tq){float*__restrict__ tr=tn[0];float ts=to[0];float*__restrict__ tt=tn[1
];float tu=to[1];float*__restrict__ tv=tn[2];float tw=to[2];float*__restrict__
tx=tn[3];float ty=to[3];float*__restrict__ tz=tn[4];float ua=to[4];float*//////
__restrict__ ub=tn[5];float uc=to[5];float*__restrict__ ud=tn[6];float ue=to[6]
;{__m128 uf=_mm_set_ps1(ts);__m128 ug=_mm_set_ps1(tu);__m128 uh=_mm_set_ps1(tw)
;__m128 ui=_mm_set_ps1(ty);__m128 uj=_mm_set_ps1(ua);__m128 uk=_mm_set_ps1(uc);
__m128 ul=_mm_set_ps1(ue);_Pragma("GCC unroll 1")_Pragma("GCC novector")while((
(char*)tq-(char*)tp)>=(16*4)){__m128 um,un,uo,up,uq,ur,us,ut;__asm__(""::"r"(tr
));(uq)=_mm_loadu_ps((float const*)(tp));(ur)=_mm_loadu_ps((float const*)(tp+4)
);(us)=_mm_loadu_ps((float const*)(tp+(2*4)));(ut)=_mm_loadu_ps((float const*)(
tp+(3*4)));(um)=_mm_loadu_ps((float const*)(tr));(un)=_mm_loadu_ps((float const
*)(tr+4));(uo)=_mm_loadu_ps((float const*)(tr+(2*4)));(up)=_mm_loadu_ps((float
const*)(tr+(3*4)));(um)=_mm_add_ps(um,_mm_mul_ps(uq,uf));(un)=_mm_add_ps(un,///
_mm_mul_ps(ur,uf));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,uf));(up)=_mm_add_ps(up,///
_mm_mul_ps(ut,uf));_mm_storeu_ps((float*)(tr),um);_mm_storeu_ps((float*)(tr+4),
un);_mm_storeu_ps((float*)(tr+(2*4)),uo);_mm_storeu_ps((float*)(tr+(3*4)),up);(
um)=_mm_loadu_ps((float const*)(tt));(un)=_mm_loadu_ps((float const*)(tt+4));(
uo)=_mm_loadu_ps((float const*)(tt+(2*4)));(up)=_mm_loadu_ps((float const*)(tt+
(3*4)));(um)=_mm_add_ps(um,_mm_mul_ps(uq,ug));(un)=_mm_add_ps(un,_mm_mul_ps(ur,
ug));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,ug));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ug)
);_mm_storeu_ps((float*)(tt),um);_mm_storeu_ps((float*)(tt+4),un);_mm_storeu_ps
((float*)(tt+(2*4)),uo);_mm_storeu_ps((float*)(tt+(3*4)),up);(um)=_mm_loadu_ps(
(float const*)(tv));(un)=_mm_loadu_ps((float const*)(tv+4));(uo)=_mm_loadu_ps((
float const*)(tv+(2*4)));(up)=_mm_loadu_ps((float const*)(tv+(3*4)));(um)=/////
_mm_add_ps(um,_mm_mul_ps(uq,uh));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uh));(uo)=///
_mm_add_ps(uo,_mm_mul_ps(us,uh));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uh));////////
_mm_storeu_ps((float*)(tv),um);_mm_storeu_ps((float*)(tv+4),un);_mm_storeu_ps((
float*)(tv+(2*4)),uo);_mm_storeu_ps((float*)(tv+(3*4)),up);(um)=_mm_loadu_ps((
float const*)(tx));(un)=_mm_loadu_ps((float const*)(tx+4));(uo)=_mm_loadu_ps((
float const*)(tx+(2*4)));(up)=_mm_loadu_ps((float const*)(tx+(3*4)));(um)=/////
_mm_add_ps(um,_mm_mul_ps(uq,ui));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ui));(uo)=///
_mm_add_ps(uo,_mm_mul_ps(us,ui));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ui));////////
_mm_storeu_ps((float*)(tx),um);_mm_storeu_ps((float*)(tx+4),un);_mm_storeu_ps((
float*)(tx+(2*4)),uo);_mm_storeu_ps((float*)(tx+(3*4)),up);(um)=_mm_loadu_ps((
float const*)(tz));(un)=_mm_loadu_ps((float const*)(tz+4));(uo)=_mm_loadu_ps((
float const*)(tz+(2*4)));(up)=_mm_loadu_ps((float const*)(tz+(3*4)));(um)=/////
_mm_add_ps(um,_mm_mul_ps(uq,uj));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uj));(uo)=///
_mm_add_ps(uo,_mm_mul_ps(us,uj));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uj));////////
_mm_storeu_ps((float*)(tz),um);_mm_storeu_ps((float*)(tz+4),un);_mm_storeu_ps((
float*)(tz+(2*4)),uo);_mm_storeu_ps((float*)(tz+(3*4)),up);(um)=_mm_loadu_ps((
float const*)(ub));(un)=_mm_loadu_ps((float const*)(ub+4));(uo)=_mm_loadu_ps((
float const*)(ub+(2*4)));(up)=_mm_loadu_ps((float const*)(ub+(3*4)));(um)=/////
_mm_add_ps(um,_mm_mul_ps(uq,uk));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uk));(uo)=///
_mm_add_ps(uo,_mm_mul_ps(us,uk));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uk));////////
_mm_storeu_ps((float*)(ub),um);_mm_storeu_ps((float*)(ub+4),un);_mm_storeu_ps((
float*)(ub+(2*4)),uo);_mm_storeu_ps((float*)(ub+(3*4)),up);(um)=_mm_loadu_ps((
float const*)(ud));(un)=_mm_loadu_ps((float const*)(ud+4));(uo)=_mm_loadu_ps((
float const*)(ud+(2*4)));(up)=_mm_loadu_ps((float const*)(ud+(3*4)));(um)=/////
_mm_add_ps(um,_mm_mul_ps(uq,ul));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ul));(uo)=///
_mm_add_ps(uo,_mm_mul_ps(us,ul));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ul));////////
_mm_storeu_ps((float*)(ud),um);_mm_storeu_ps((float*)(ud+4),un);_mm_storeu_ps((
float*)(ud+(2*4)),uo);_mm_storeu_ps((float*)(ud+(3*4)),up);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);ud+=(4*4);}_Pragma(//////////
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128//
uu,uv;__asm__(""::"r"(tr));(uv)=_mm_loadu_ps((float const*)(tp));(uu)=/////////
_mm_loadu_ps((float const*)(tr));(uu)=_mm_add_ps(uu,_mm_mul_ps(uv,(uf)));//////
_mm_storeu_ps((float*)(tr),uu);(uu)=_mm_loadu_ps((float const*)(tt));(uu)=/////
_mm_add_ps(uu,_mm_mul_ps(uv,(ug)));_mm_storeu_ps((float*)(tt),uu);(uu)=////////
_mm_loadu_ps((float const*)(tv));(uu)=_mm_add_ps(uu,_mm_mul_ps(uv,(uh)));//////
_mm_storeu_ps((float*)(tv),uu);(uu)=_mm_loadu_ps((float const*)(tx));(uu)=/////
_mm_add_ps(uu,_mm_mul_ps(uv,(ui)));_mm_storeu_ps((float*)(tx),uu);(uu)=////////
_mm_loadu_ps((float const*)(tz));(uu)=_mm_add_ps(uu,_mm_mul_ps(uv,(uj)));//////
_mm_storeu_ps((float*)(tz),uu);(uu)=_mm_loadu_ps((float const*)(ub));(uu)=/////
_mm_add_ps(uu,_mm_mul_ps(uv,(uk)));_mm_storeu_ps((float*)(ub),uu);(uu)=////////
_mm_loadu_ps((float const*)(ud));(uu)=_mm_add_ps(uu,_mm_mul_ps(uv,(ul)));//////
_mm_storeu_ps((float*)(ud),uu);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;ud+=4;
}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uw=tp[0];///
__asm__(""::"r"(tr));tr[0]+=(uw*ts);tt[0]+=(uw*tu);tv[0]+=(uw*tw);tx[0]+=(uw*ty
);tz[0]+=(uw*ua);ub[0]+=(uw*uc);ud[0]+=(uw*ue);++tp;++tr;++tt;++tv;++tx;++tz;++
ub;++ud;}}static void ro(float*tn,float const*to,float const**tp,float const*tq
){float*__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=
tp[1];float tv=to[1];float const*tw=tp[2];float tx=to[2];float const*ty=tp[3];
float tz=to[3];float const*ua=tp[4];float ub=to[4];float const*uc=tp[5];float//
ud=to[5];float const*ue=tp[6];float uf=to[6];{__m128 ug=_mm_set_ps1(tt);__m128
uh=_mm_set_ps1(tv);__m128 ui=_mm_set_ps1(tx);__m128 uj=_mm_set_ps1(tz);__m128//
uk=_mm_set_ps1(ub);__m128 ul=_mm_set_ps1(ud);__m128 um=_mm_set_ps1(uf);_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){////
__m128 un,uo,up,uq,ur,us,ut,uu;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16
*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((//
char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);///
_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(uc+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ue+(16*4)),_MM_HINT_T0);(un)=_mm_loadu_ps((//
float const*)(tr));(uo)=_mm_loadu_ps((float const*)(tr+4));(up)=_mm_loadu_ps((
float const*)(tr+(2*4)));(uq)=_mm_loadu_ps((float const*)(tr+(3*4)));(ur)=/////
_mm_loadu_ps((float const*)(ts));(us)=_mm_loadu_ps((float const*)(ts+4));(ut)=
_mm_loadu_ps((float const*)(ts+(2*4)));(uu)=_mm_loadu_ps((float const*)(ts+(3*4
)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ug));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,ug))
;(up)=_mm_add_ps(up,_mm_mul_ps(ut,ug));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ug));(
ur)=_mm_loadu_ps((float const*)(tu));(us)=_mm_loadu_ps((float const*)(tu+4));(
ut)=_mm_loadu_ps((float const*)(tu+(2*4)));(uu)=_mm_loadu_ps((float const*)(tu+
(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uh));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,
uh));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uh));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uh)
);(ur)=_mm_loadu_ps((float const*)(tw));(us)=_mm_loadu_ps((float const*)(tw+4))
;(ut)=_mm_loadu_ps((float const*)(tw+(2*4)));(uu)=_mm_loadu_ps((float const*)(
tw+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ui));(uo)=_mm_add_ps(uo,_mm_mul_ps(
us,ui));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ui));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,
ui));(ur)=_mm_loadu_ps((float const*)(ty));(us)=_mm_loadu_ps((float const*)(ty+
4));(ut)=_mm_loadu_ps((float const*)(ty+(2*4)));(uu)=_mm_loadu_ps((float const*
)(ty+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uj));(uo)=_mm_add_ps(uo,/////////
_mm_mul_ps(us,uj));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uj));(uq)=_mm_add_ps(uq,///
_mm_mul_ps(uu,uj));(ur)=_mm_loadu_ps((float const*)(ua));(us)=_mm_loadu_ps((///
float const*)(ua+4));(ut)=_mm_loadu_ps((float const*)(ua+(2*4)));(uu)=/////////
_mm_loadu_ps((float const*)(ua+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,uk));(
uo)=_mm_add_ps(uo,_mm_mul_ps(us,uk));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uk));(uq)
=_mm_add_ps(uq,_mm_mul_ps(uu,uk));(ur)=_mm_loadu_ps((float const*)(uc));(us)=//
_mm_loadu_ps((float const*)(uc+4));(ut)=_mm_loadu_ps((float const*)(uc+(2*4)));
(uu)=_mm_loadu_ps((float const*)(uc+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,ul
));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,ul));(up)=_mm_add_ps(up,_mm_mul_ps(ut,ul));
(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ul));(ur)=_mm_loadu_ps((float const*)(ue));(us
)=_mm_loadu_ps((float const*)(ue+4));(ut)=_mm_loadu_ps((float const*)(ue+(2*4))
);(uu)=_mm_loadu_ps((float const*)(ue+(3*4)));(un)=_mm_add_ps(un,_mm_mul_ps(ur,
um));(uo)=_mm_add_ps(uo,_mm_mul_ps(us,um));(up)=_mm_add_ps(up,_mm_mul_ps(ut,um)
);(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,um));_mm_storeu_ps((float*)(tr),un);////////
_mm_storeu_ps((float*)(tr+4),uo);_mm_storeu_ps((float*)(tr+(2*4)),up);/////////
_mm_storeu_ps((float*)(tr+(3*4)),uq);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty
+=(4*4);ua+=(4*4);uc+=(4*4);ue+=(4*4);}_Pragma("GCC unroll 1")_Pragma(/////////
"GCC novector")while(((char*)tq-(char*)ts)>=16){__m128 uv,uw;__asm__(""::"r"(tr
));(uv)=_mm_loadu_ps((float const*)(tr));(uw)=_mm_loadu_ps((float const*)(ts));
(uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(ug)));(uw)=_mm_loadu_ps((float const*)(tu));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(uh)));(uw)=_mm_loadu_ps((float const*)(tw));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(ui)));(uw)=_mm_loadu_ps((float const*)(ty));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(uj)));(uw)=_mm_loadu_ps((float const*)(ua));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(uk)));(uw)=_mm_loadu_ps((float const*)(uc));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(ul)));(uw)=_mm_loadu_ps((float const*)(ue));(
uv)=_mm_add_ps(uv,_mm_mul_ps(uw,(um)));_mm_storeu_ps((float*)(tr),uv);tr+=4;ts
+=4;tu+=4;tw+=4;ty+=4;ua+=4;uc+=4;ue+=4;}}_Pragma("GCC unroll 1")_Pragma(//////
"GCC novector")while(ts<tq){float ux;__asm__(""::"r"(tr));ux=tr[0]+ts[0]*tt;ux
+=tu[0]*tv;ux+=tw[0]*tx;ux+=ty[0]*tz;ux+=ua[0]*ub;ux+=uc[0]*ud;ux+=ue[0]*uf;tr[
0]=ux;++tr;++ts;++tu;++tw;++ty;++ua;++uc;++ue;}}static void rp(float**tn,float
const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];float ts=to
[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv=tn[2];////
float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3];float*__restrict__ tz
=tn[4];float ua=to[4];float*__restrict__ ub=tn[5];float uc=to[5];float*////////
__restrict__ ud=tn[6];float ue=to[6];float*__restrict__ uf=tn[7];float ug=to[7]
;{__m128 uh=_mm_set_ps1(ts);__m128 ui=_mm_set_ps1(tu);__m128 uj=_mm_set_ps1(tw)
;__m128 uk=_mm_set_ps1(ty);__m128 ul=_mm_set_ps1(ua);__m128 um=_mm_set_ps1(uc);
__m128 un=_mm_set_ps1(ue);__m128 uo=_mm_set_ps1(ug);_Pragma("GCC unroll 1")////
_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){__m128 up,uq,ur,us,
ut,uu,uv,uw;__asm__(""::"r"(tr));(ut)=_mm_loadu_ps((float const*)(tp));(uu)=///
_mm_loadu_ps((float const*)(tp+4));(uv)=_mm_loadu_ps((float const*)(tp+(2*4)));
(uw)=_mm_loadu_ps((float const*)(tp+(3*4)));(up)=_mm_mul_ps(ut,uh);(uq)=///////
_mm_mul_ps(uu,uh);(ur)=_mm_mul_ps(uv,uh);(us)=_mm_mul_ps(uw,uh);_mm_storeu_ps((
float*)(tr),up);_mm_storeu_ps((float*)(tr+4),uq);_mm_storeu_ps((float*)(tr+(2*4
)),ur);_mm_storeu_ps((float*)(tr+(3*4)),us);(up)=_mm_mul_ps(ut,ui);(uq)=///////
_mm_mul_ps(uu,ui);(ur)=_mm_mul_ps(uv,ui);(us)=_mm_mul_ps(uw,ui);_mm_storeu_ps((
float*)(tt),up);_mm_storeu_ps((float*)(tt+4),uq);_mm_storeu_ps((float*)(tt+(2*4
)),ur);_mm_storeu_ps((float*)(tt+(3*4)),us);(up)=_mm_mul_ps(ut,uj);(uq)=///////
_mm_mul_ps(uu,uj);(ur)=_mm_mul_ps(uv,uj);(us)=_mm_mul_ps(uw,uj);_mm_storeu_ps((
float*)(tv),up);_mm_storeu_ps((float*)(tv+4),uq);_mm_storeu_ps((float*)(tv+(2*4
)),ur);_mm_storeu_ps((float*)(tv+(3*4)),us);(up)=_mm_mul_ps(ut,uk);(uq)=///////
_mm_mul_ps(uu,uk);(ur)=_mm_mul_ps(uv,uk);(us)=_mm_mul_ps(uw,uk);_mm_storeu_ps((
float*)(tx),up);_mm_storeu_ps((float*)(tx+4),uq);_mm_storeu_ps((float*)(tx+(2*4
)),ur);_mm_storeu_ps((float*)(tx+(3*4)),us);(up)=_mm_mul_ps(ut,ul);(uq)=///////
_mm_mul_ps(uu,ul);(ur)=_mm_mul_ps(uv,ul);(us)=_mm_mul_ps(uw,ul);_mm_storeu_ps((
float*)(tz),up);_mm_storeu_ps((float*)(tz+4),uq);_mm_storeu_ps((float*)(tz+(2*4
)),ur);_mm_storeu_ps((float*)(tz+(3*4)),us);(up)=_mm_mul_ps(ut,um);(uq)=///////
_mm_mul_ps(uu,um);(ur)=_mm_mul_ps(uv,um);(us)=_mm_mul_ps(uw,um);_mm_storeu_ps((
float*)(ub),up);_mm_storeu_ps((float*)(ub+4),uq);_mm_storeu_ps((float*)(ub+(2*4
)),ur);_mm_storeu_ps((float*)(ub+(3*4)),us);(up)=_mm_mul_ps(ut,un);(uq)=///////
_mm_mul_ps(uu,un);(ur)=_mm_mul_ps(uv,un);(us)=_mm_mul_ps(uw,un);_mm_storeu_ps((
float*)(ud),up);_mm_storeu_ps((float*)(ud+4),uq);_mm_storeu_ps((float*)(ud+(2*4
)),ur);_mm_storeu_ps((float*)(ud+(3*4)),us);(up)=_mm_mul_ps(ut,uo);(uq)=///////
_mm_mul_ps(uu,uo);(ur)=_mm_mul_ps(uv,uo);(us)=_mm_mul_ps(uw,uo);_mm_storeu_ps((
float*)(uf),up);_mm_storeu_ps((float*)(uf+4),uq);_mm_storeu_ps((float*)(uf+(2*4
)),ur);_mm_storeu_ps((float*)(uf+(3*4)),us);tp+=(4*4);tr+=(4*4);tt+=(4*4);tv+=(
4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);ud+=(4*4);uf+=(4*4);}_Pragma("GCC unroll 1")
_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128 ux,uy;__asm__(""
::"r"(tr));(uy)=_mm_loadu_ps((float const*)(tp));(ux)=_mm_mul_ps(uy,(uh));/////
_mm_storeu_ps((float*)(tr),ux);(ux)=_mm_mul_ps(uy,(ui));_mm_storeu_ps((float*)(
tt),ux);(ux)=_mm_mul_ps(uy,(uj));_mm_storeu_ps((float*)(tv),ux);(ux)=_mm_mul_ps
(uy,(uk));_mm_storeu_ps((float*)(tx),ux);(ux)=_mm_mul_ps(uy,(ul));_mm_storeu_ps
((float*)(tz),ux);(ux)=_mm_mul_ps(uy,(um));_mm_storeu_ps((float*)(ub),ux);(ux)=
_mm_mul_ps(uy,(un));_mm_storeu_ps((float*)(ud),ux);(ux)=_mm_mul_ps(uy,(uo));///
_mm_storeu_ps((float*)(uf),ux);tp+=4;tr+=4;tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;ud+=4;
uf+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(tp<tq){float uz=tp[
0];__asm__(""::"r"(tr));tr[0]=(uz*ts);tt[0]=(uz*tu);tv[0]=(uz*tw);tx[0]=(uz*ty)
;tz[0]=(uz*ua);ub[0]=(uz*uc);ud[0]=(uz*ue);uf[0]=(uz*ug);++tp;++tr;++tt;++tv;++
tx;++tz;++ub;++ud;++uf;}}static void rq(float*tn,float const*to,float const**tp
,float const*tq){float*__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];
float const*tu=tp[1];float tv=to[1];float const*tw=tp[2];float tx=to[2];float//
const*ty=tp[3];float tz=to[3];float const*ua=tp[4];float ub=to[4];float const*
uc=tp[5];float ud=to[5];float const*ue=tp[6];float uf=to[6];float const*ug=tp[7
];float uh=to[7];{__m128 ui=_mm_set_ps1(tt);__m128 uj=_mm_set_ps1(tv);__m128 uk
=_mm_set_ps1(tx);__m128 ul=_mm_set_ps1(tz);__m128 um=_mm_set_ps1(ub);__m128 un=
_mm_set_ps1(ud);__m128 uo=_mm_set_ps1(uf);__m128 up=_mm_set_ps1(uh);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){////
__m128 uq,ur,us,ut,uu,uv,uw,ux;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16
*4)),_MM_HINT_T0);_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((//
char*)(tw+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);///
_mm_prefetch((char*)(ua+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(uc+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ue+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(
ug+(16*4)),_MM_HINT_T0);(uu)=_mm_loadu_ps((float const*)(ts));(uv)=_mm_loadu_ps
((float const*)(ts+4));(uw)=_mm_loadu_ps((float const*)(ts+(2*4)));(ux)=///////
_mm_loadu_ps((float const*)(ts+(3*4)));(uq)=_mm_mul_ps(uu,ui);(ur)=_mm_mul_ps(
uv,ui);(us)=_mm_mul_ps(uw,ui);(ut)=_mm_mul_ps(ux,ui);(uu)=_mm_loadu_ps((float//
const*)(tu));(uv)=_mm_loadu_ps((float const*)(tu+4));(uw)=_mm_loadu_ps((float//
const*)(tu+(2*4)));(ux)=_mm_loadu_ps((float const*)(tu+(3*4)));(uq)=_mm_add_ps(
uq,_mm_mul_ps(uu,uj));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,uj));(us)=_mm_add_ps(us,
_mm_mul_ps(uw,uj));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,uj));(uu)=_mm_loadu_ps((///
float const*)(tw));(uv)=_mm_loadu_ps((float const*)(tw+4));(uw)=_mm_loadu_ps((
float const*)(tw+(2*4)));(ux)=_mm_loadu_ps((float const*)(tw+(3*4)));(uq)=/////
_mm_add_ps(uq,_mm_mul_ps(uu,uk));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,uk));(us)=///
_mm_add_ps(us,_mm_mul_ps(uw,uk));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,uk));(uu)=///
_mm_loadu_ps((float const*)(ty));(uv)=_mm_loadu_ps((float const*)(ty+4));(uw)=
_mm_loadu_ps((float const*)(ty+(2*4)));(ux)=_mm_loadu_ps((float const*)(ty+(3*4
)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ul));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,ul))
;(us)=_mm_add_ps(us,_mm_mul_ps(uw,ul));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,ul));(
uu)=_mm_loadu_ps((float const*)(ua));(uv)=_mm_loadu_ps((float const*)(ua+4));(
uw)=_mm_loadu_ps((float const*)(ua+(2*4)));(ux)=_mm_loadu_ps((float const*)(ua+
(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,um));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,
um));(us)=_mm_add_ps(us,_mm_mul_ps(uw,um));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,um)
);(uu)=_mm_loadu_ps((float const*)(uc));(uv)=_mm_loadu_ps((float const*)(uc+4))
;(uw)=_mm_loadu_ps((float const*)(uc+(2*4)));(ux)=_mm_loadu_ps((float const*)(
uc+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,un));(ur)=_mm_add_ps(ur,_mm_mul_ps(
uv,un));(us)=_mm_add_ps(us,_mm_mul_ps(uw,un));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,
un));(uu)=_mm_loadu_ps((float const*)(ue));(uv)=_mm_loadu_ps((float const*)(ue+
4));(uw)=_mm_loadu_ps((float const*)(ue+(2*4)));(ux)=_mm_loadu_ps((float const*
)(ue+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uo));(ur)=_mm_add_ps(ur,/////////
_mm_mul_ps(uv,uo));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uo));(ut)=_mm_add_ps(ut,///
_mm_mul_ps(ux,uo));(uu)=_mm_loadu_ps((float const*)(ug));(uv)=_mm_loadu_ps((///
float const*)(ug+4));(uw)=_mm_loadu_ps((float const*)(ug+(2*4)));(ux)=/////////
_mm_loadu_ps((float const*)(ug+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,up));(
ur)=_mm_add_ps(ur,_mm_mul_ps(uv,up));(us)=_mm_add_ps(us,_mm_mul_ps(uw,up));(ut)
=_mm_add_ps(ut,_mm_mul_ps(ux,up));_mm_storeu_ps((float*)(tr),uq);_mm_storeu_ps(
(float*)(tr+4),ur);_mm_storeu_ps((float*)(tr+(2*4)),us);_mm_storeu_ps((float*)(
tr+(3*4)),ut);tr+=(4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);uc+=(
4*4);ue+=(4*4);ug+=(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while((
(char*)tq-(char*)ts)>=16){__m128 uy,uz;__asm__(""::"r"(tr));(uz)=_mm_loadu_ps((
float const*)(ts));(uy)=_mm_mul_ps(uz,(ui));(uz)=_mm_loadu_ps((float const*)(tu
));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uj)));(uz)=_mm_loadu_ps((float const*)(tw)
);(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uk)));(uz)=_mm_loadu_ps((float const*)(ty))
;(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(ul)));(uz)=_mm_loadu_ps((float const*)(ua));
(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(um)));(uz)=_mm_loadu_ps((float const*)(uc));(
uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(un)));(uz)=_mm_loadu_ps((float const*)(ue));(
uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uo)));(uz)=_mm_loadu_ps((float const*)(ug));(
uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(up)));_mm_storeu_ps((float*)(tr),uy);tr+=4;ts
+=4;tu+=4;tw+=4;ty+=4;ua+=4;uc+=4;ue+=4;ug+=4;}}_Pragma("GCC unroll 1")_Pragma(
"GCC novector")while(ts<tq){float va;__asm__(""::"r"(tr));va=ts[0]*tt;va+=tu[0]
*tv;va+=tw[0]*tx;va+=ty[0]*tz;va+=ua[0]*ub;va+=uc[0]*ud;va+=ue[0]*uf;va+=ug[0]*
uh;tr[0]=va;++tr;++ts;++tu;++tw;++ty;++ua;++uc;++ue;++ug;}}static void rr(float
**tn,float const*to,float const*tp,float const*tq){float*__restrict__ tr=tn[0];
float ts=to[0];float*__restrict__ tt=tn[1];float tu=to[1];float*__restrict__ tv
=tn[2];float tw=to[2];float*__restrict__ tx=tn[3];float ty=to[3];float*////////
__restrict__ tz=tn[4];float ua=to[4];float*__restrict__ ub=tn[5];float uc=to[5]
;float*__restrict__ ud=tn[6];float ue=to[6];float*__restrict__ uf=tn[7];float//
ug=to[7];{__m128 uh=_mm_set_ps1(ts);__m128 ui=_mm_set_ps1(tu);__m128 uj=///////
_mm_set_ps1(tw);__m128 uk=_mm_set_ps1(ty);__m128 ul=_mm_set_ps1(ua);__m128 um=
_mm_set_ps1(uc);__m128 un=_mm_set_ps1(ue);__m128 uo=_mm_set_ps1(ug);_Pragma(///
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=(16*4)){////
__m128 up,uq,ur,us,ut,uu,uv,uw;__asm__(""::"r"(tr));(ut)=_mm_loadu_ps((float///
const*)(tp));(uu)=_mm_loadu_ps((float const*)(tp+4));(uv)=_mm_loadu_ps((float//
const*)(tp+(2*4)));(uw)=_mm_loadu_ps((float const*)(tp+(3*4)));(up)=///////////
_mm_loadu_ps((float const*)(tr));(uq)=_mm_loadu_ps((float const*)(tr+4));(ur)=
_mm_loadu_ps((float const*)(tr+(2*4)));(us)=_mm_loadu_ps((float const*)(tr+(3*4
)));(up)=_mm_add_ps(up,_mm_mul_ps(ut,uh));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uh))
;(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,uh));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uh));//
_mm_storeu_ps((float*)(tr),up);_mm_storeu_ps((float*)(tr+4),uq);_mm_storeu_ps((
float*)(tr+(2*4)),ur);_mm_storeu_ps((float*)(tr+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(tt));(uq)=_mm_loadu_ps((float const*)(tt+4));(ur)=_mm_loadu_ps((
float const*)(tt+(2*4)));(us)=_mm_loadu_ps((float const*)(tt+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,ui));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ui));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,ui));(us)=_mm_add_ps(us,_mm_mul_ps(uw,ui));////////
_mm_storeu_ps((float*)(tt),up);_mm_storeu_ps((float*)(tt+4),uq);_mm_storeu_ps((
float*)(tt+(2*4)),ur);_mm_storeu_ps((float*)(tt+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(tv));(uq)=_mm_loadu_ps((float const*)(tv+4));(ur)=_mm_loadu_ps((
float const*)(tv+(2*4)));(us)=_mm_loadu_ps((float const*)(tv+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,uj));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uj));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,uj));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uj));////////
_mm_storeu_ps((float*)(tv),up);_mm_storeu_ps((float*)(tv+4),uq);_mm_storeu_ps((
float*)(tv+(2*4)),ur);_mm_storeu_ps((float*)(tv+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(tx));(uq)=_mm_loadu_ps((float const*)(tx+4));(ur)=_mm_loadu_ps((
float const*)(tx+(2*4)));(us)=_mm_loadu_ps((float const*)(tx+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,uk));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uk));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,uk));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uk));////////
_mm_storeu_ps((float*)(tx),up);_mm_storeu_ps((float*)(tx+4),uq);_mm_storeu_ps((
float*)(tx+(2*4)),ur);_mm_storeu_ps((float*)(tx+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(tz));(uq)=_mm_loadu_ps((float const*)(tz+4));(ur)=_mm_loadu_ps((
float const*)(tz+(2*4)));(us)=_mm_loadu_ps((float const*)(tz+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,ul));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ul));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,ul));(us)=_mm_add_ps(us,_mm_mul_ps(uw,ul));////////
_mm_storeu_ps((float*)(tz),up);_mm_storeu_ps((float*)(tz+4),uq);_mm_storeu_ps((
float*)(tz+(2*4)),ur);_mm_storeu_ps((float*)(tz+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(ub));(uq)=_mm_loadu_ps((float const*)(ub+4));(ur)=_mm_loadu_ps((
float const*)(ub+(2*4)));(us)=_mm_loadu_ps((float const*)(ub+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,um));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,um));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,um));(us)=_mm_add_ps(us,_mm_mul_ps(uw,um));////////
_mm_storeu_ps((float*)(ub),up);_mm_storeu_ps((float*)(ub+4),uq);_mm_storeu_ps((
float*)(ub+(2*4)),ur);_mm_storeu_ps((float*)(ub+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(ud));(uq)=_mm_loadu_ps((float const*)(ud+4));(ur)=_mm_loadu_ps((
float const*)(ud+(2*4)));(us)=_mm_loadu_ps((float const*)(ud+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,un));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,un));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,un));(us)=_mm_add_ps(us,_mm_mul_ps(uw,un));////////
_mm_storeu_ps((float*)(ud),up);_mm_storeu_ps((float*)(ud+4),uq);_mm_storeu_ps((
float*)(ud+(2*4)),ur);_mm_storeu_ps((float*)(ud+(3*4)),us);(up)=_mm_loadu_ps((
float const*)(uf));(uq)=_mm_loadu_ps((float const*)(uf+4));(ur)=_mm_loadu_ps((
float const*)(uf+(2*4)));(us)=_mm_loadu_ps((float const*)(uf+(3*4)));(up)=/////
_mm_add_ps(up,_mm_mul_ps(ut,uo));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uo));(ur)=///
_mm_add_ps(ur,_mm_mul_ps(uv,uo));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uo));////////
_mm_storeu_ps((float*)(uf),up);_mm_storeu_ps((float*)(uf+4),uq);_mm_storeu_ps((
float*)(uf+(2*4)),ur);_mm_storeu_ps((float*)(uf+(3*4)),us);tp+=(4*4);tr+=(4*4);
tt+=(4*4);tv+=(4*4);tx+=(4*4);tz+=(4*4);ub+=(4*4);ud+=(4*4);uf+=(4*4);}_Pragma(
"GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)tp)>=16){__m128//
ux,uy;__asm__(""::"r"(tr));(uy)=_mm_loadu_ps((float const*)(tp));(ux)=/////////
_mm_loadu_ps((float const*)(tr));(ux)=_mm_add_ps(ux,_mm_mul_ps(uy,(uh)));//////
_mm_storeu_ps((float*)(tr),ux);(ux)=_mm_loadu_ps((float const*)(tt));(ux)=/////
_mm_add_ps(ux,_mm_mul_ps(uy,(ui)));_mm_storeu_ps((float*)(tt),ux);(ux)=////////
_mm_loadu_ps((float const*)(tv));(ux)=_mm_add_ps(ux,_mm_mul_ps(uy,(uj)));//////
_mm_storeu_ps((float*)(tv),ux);(ux)=_mm_loadu_ps((float const*)(tx));(ux)=/////
_mm_add_ps(ux,_mm_mul_ps(uy,(uk)));_mm_storeu_ps((float*)(tx),ux);(ux)=////////
_mm_loadu_ps((float const*)(tz));(ux)=_mm_add_ps(ux,_mm_mul_ps(uy,(ul)));//////
_mm_storeu_ps((float*)(tz),ux);(ux)=_mm_loadu_ps((float const*)(ub));(ux)=/////
_mm_add_ps(ux,_mm_mul_ps(uy,(um)));_mm_storeu_ps((float*)(ub),ux);(ux)=////////
_mm_loadu_ps((float const*)(ud));(ux)=_mm_add_ps(ux,_mm_mul_ps(uy,(un)));//////
_mm_storeu_ps((float*)(ud),ux);(ux)=_mm_loadu_ps((float const*)(uf));(ux)=/////
_mm_add_ps(ux,_mm_mul_ps(uy,(uo)));_mm_storeu_ps((float*)(uf),ux);tp+=4;tr+=4;
tt+=4;tv+=4;tx+=4;tz+=4;ub+=4;ud+=4;uf+=4;}}_Pragma("GCC unroll 1")_Pragma(////
"GCC novector")while(tp<tq){float uz=tp[0];__asm__(""::"r"(tr));tr[0]+=(uz*ts);
tt[0]+=(uz*tu);tv[0]+=(uz*tw);tx[0]+=(uz*ty);tz[0]+=(uz*ua);ub[0]+=(uz*uc);ud[0
]+=(uz*ue);uf[0]+=(uz*ug);++tp;++tr;++tt;++tv;++tx;++tz;++ub;++ud;++uf;}}static
void rt(float*tn,float const*to,float const**tp,float const*tq){float*/////////
__restrict__ tr=tn;float const*ts=tp[0];float tt=to[0];float const*tu=tp[1];///
float tv=to[1];float const*tw=tp[2];float tx=to[2];float const*ty=tp[3];float//
tz=to[3];float const*ua=tp[4];float ub=to[4];float const*uc=tp[5];float ud=to[5
];float const*ue=tp[6];float uf=to[6];float const*ug=tp[7];float uh=to[7];{////
__m128 ui=_mm_set_ps1(tt);__m128 uj=_mm_set_ps1(tv);__m128 uk=_mm_set_ps1(tx);
__m128 ul=_mm_set_ps1(tz);__m128 um=_mm_set_ps1(ub);__m128 un=_mm_set_ps1(ud);
__m128 uo=_mm_set_ps1(uf);__m128 up=_mm_set_ps1(uh);_Pragma("GCC unroll 1")////
_Pragma("GCC novector")while(((char*)tq-(char*)ts)>=(16*4)){__m128 uq,ur,us,ut,
uu,uv,uw,ux;__asm__(""::"r"(tr));_mm_prefetch((char*)(ts+(16*4)),_MM_HINT_T0);
_mm_prefetch((char*)(tu+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(tw+(16*4)),//
_MM_HINT_T0);_mm_prefetch((char*)(ty+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(
ua+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(uc+(16*4)),_MM_HINT_T0);//////////
_mm_prefetch((char*)(ue+(16*4)),_MM_HINT_T0);_mm_prefetch((char*)(ug+(16*4)),//
_MM_HINT_T0);(uq)=_mm_loadu_ps((float const*)(tr));(ur)=_mm_loadu_ps((float////
const*)(tr+4));(us)=_mm_loadu_ps((float const*)(tr+(2*4)));(ut)=_mm_loadu_ps((
float const*)(tr+(3*4)));(uu)=_mm_loadu_ps((float const*)(ts));(uv)=///////////
_mm_loadu_ps((float const*)(ts+4));(uw)=_mm_loadu_ps((float const*)(ts+(2*4)));
(ux)=_mm_loadu_ps((float const*)(ts+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,ui
));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,ui));(us)=_mm_add_ps(us,_mm_mul_ps(uw,ui));
(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,ui));(uu)=_mm_loadu_ps((float const*)(tu));(uv
)=_mm_loadu_ps((float const*)(tu+4));(uw)=_mm_loadu_ps((float const*)(tu+(2*4))
);(ux)=_mm_loadu_ps((float const*)(tu+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,
uj));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,uj));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uj)
);(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,uj));(uu)=_mm_loadu_ps((float const*)(tw));(
uv)=_mm_loadu_ps((float const*)(tw+4));(uw)=_mm_loadu_ps((float const*)(tw+(2*4
)));(ux)=_mm_loadu_ps((float const*)(tw+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(
uu,uk));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,uk));(us)=_mm_add_ps(us,_mm_mul_ps(uw,
uk));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,uk));(uu)=_mm_loadu_ps((float const*)(ty)
);(uv)=_mm_loadu_ps((float const*)(ty+4));(uw)=_mm_loadu_ps((float const*)(ty+(
2*4)));(ux)=_mm_loadu_ps((float const*)(ty+(3*4)));(uq)=_mm_add_ps(uq,/////////
_mm_mul_ps(uu,ul));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,ul));(us)=_mm_add_ps(us,///
_mm_mul_ps(uw,ul));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,ul));(uu)=_mm_loadu_ps((///
float const*)(ua));(uv)=_mm_loadu_ps((float const*)(ua+4));(uw)=_mm_loadu_ps((
float const*)(ua+(2*4)));(ux)=_mm_loadu_ps((float const*)(ua+(3*4)));(uq)=/////
_mm_add_ps(uq,_mm_mul_ps(uu,um));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,um));(us)=///
_mm_add_ps(us,_mm_mul_ps(uw,um));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,um));(uu)=///
_mm_loadu_ps((float const*)(uc));(uv)=_mm_loadu_ps((float const*)(uc+4));(uw)=
_mm_loadu_ps((float const*)(uc+(2*4)));(ux)=_mm_loadu_ps((float const*)(uc+(3*4
)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,un));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,un))
;(us)=_mm_add_ps(us,_mm_mul_ps(uw,un));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,un));(
uu)=_mm_loadu_ps((float const*)(ue));(uv)=_mm_loadu_ps((float const*)(ue+4));(
uw)=_mm_loadu_ps((float const*)(ue+(2*4)));(ux)=_mm_loadu_ps((float const*)(ue+
(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,uo));(ur)=_mm_add_ps(ur,_mm_mul_ps(uv,
uo));(us)=_mm_add_ps(us,_mm_mul_ps(uw,uo));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,uo)
);(uu)=_mm_loadu_ps((float const*)(ug));(uv)=_mm_loadu_ps((float const*)(ug+4))
;(uw)=_mm_loadu_ps((float const*)(ug+(2*4)));(ux)=_mm_loadu_ps((float const*)(
ug+(3*4)));(uq)=_mm_add_ps(uq,_mm_mul_ps(uu,up));(ur)=_mm_add_ps(ur,_mm_mul_ps(
uv,up));(us)=_mm_add_ps(us,_mm_mul_ps(uw,up));(ut)=_mm_add_ps(ut,_mm_mul_ps(ux,
up));_mm_storeu_ps((float*)(tr),uq);_mm_storeu_ps((float*)(tr+4),ur);//////////
_mm_storeu_ps((float*)(tr+(2*4)),us);_mm_storeu_ps((float*)(tr+(3*4)),ut);tr+=(
4*4);ts+=(4*4);tu+=(4*4);tw+=(4*4);ty+=(4*4);ua+=(4*4);uc+=(4*4);ue+=(4*4);ug+=
(4*4);}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(((char*)tq-(char*)ts
)>=16){__m128 uy,uz;__asm__(""::"r"(tr));(uy)=_mm_loadu_ps((float const*)(tr));
(uz)=_mm_loadu_ps((float const*)(ts));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(ui)));(
uz)=_mm_loadu_ps((float const*)(tu));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uj)));(
uz)=_mm_loadu_ps((float const*)(tw));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uk)));(
uz)=_mm_loadu_ps((float const*)(ty));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(ul)));(
uz)=_mm_loadu_ps((float const*)(ua));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(um)));(
uz)=_mm_loadu_ps((float const*)(uc));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(un)));(
uz)=_mm_loadu_ps((float const*)(ue));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(uo)));(
uz)=_mm_loadu_ps((float const*)(ug));(uy)=_mm_add_ps(uy,_mm_mul_ps(uz,(up)));//
_mm_storeu_ps((float*)(tr),uy);tr+=4;ts+=4;tu+=4;tw+=4;ty+=4;ua+=4;uc+=4;ue+=4;
ug+=4;}}_Pragma("GCC unroll 1")_Pragma("GCC novector")while(ts<tq){float va;///
__asm__(""::"r"(tr));va=tr[0]+ts[0]*tt;va+=tu[0]*tv;va+=tw[0]*tx;va+=ty[0]*tz;
va+=ua[0]*ub;va+=uc[0]*ud;va+=ue[0]*uf;va+=ug[0]*uh;tr[0]=va;++tr;++ts;++tu;++
tw;++ty;++ua;++uc;++ue;++ug;}}typedef void ru(float*output,float const*coeffs,
float const**inputs,float const*input0_end);static ru*rv[8]={qn,qr,qv,qz,rd,rh,
rm,rq};static ru*rx[8]={qp,qt,qx,rb,rf,rk,ro,rt};typedef void ry(float**outputs
,float const*coeffs,float const*input,float const*input_end);static ry*rz[8]={
qm,qq,qu,qy,rc,rg,rl,rp};static ry*sa[8]={qo,qs,qw,ra,re,rj,rn,rr};static void
sb(fz const*tn,void*to,float*tp,int tq){int tr=tn->l.ad.u;int ts=tn->bd;int tt=
tr*ts;void*tu;if(tn->at){;tn->at(tp,tt);;}tu=to;if(tn->am)tu=tp;;tn->au(tu,tt,
tp);;if(tn->am)tn->am(tu,tr,tq,tn->al);}static float*sc(fz const*tn,gk const*to
,int tp){assert(tp<tn->ah);return(float*)(((char*)to->ai)+(tp*tn->ag));}static
float*se(fz const*tn,gk const*to,int tp){int tq=(to->ac+(tp-to->u))%tn->ah;////
return sc(tn,to,tq);}static void sf(fz const*tn,float*to,float const*tp){float
const*tq=tp-(tn->an.l.l*tn->be);;if((tn->l.ag==STBIR_FILTER_POINT_SAMPLE)&&(tn
->l.ad.ab==1.0f))hx(to,tp,tn->l.ad.u*sizeof(float)*tn->be);else tn->as(to,tn->l
.ad.u,tq,tn->l.l,tn->l.u,tn->l.ak);;}static void sg(fz const*tn,gk*to,int tp,//
int tq,int tr,float const*ts){float*tt=to->aj;float*tu=to->l;int tv=tn->bc;int
tw=(tv)?(tn->an.l.u-tn->an.l.l+1):tn->l.ad.u;int tx=tn->be*tw;assert(tn->u.ar);
;{int ty=0,tz=tr-tq+1;assert(tz>0);do{float const*ua[8];int ub,uc=tz;if(uc>8)uc
=8;for(ub=0;ub<uc;ub++)ua[ub]=se(tn,to,ty+ub+tq);((ty==0)?rv:rx)[uc-1]((tv)?tu:
tt,ts+ty,ua,ua[0]+tx);ty+=uc;tz-=uc;}while(tz);};if(tv){tu[tx]=0.0f;tu[tx+1]=//
0.0f;sf(tn,tt,tu);}sb(tn,((char*)tn->ac)+((size_t)tp*(size_t)tn->ae),tt,tp);}//
static void sh(fz const*tn,gk*to,int tp){int tq;float*tr;mv(tn,tp,to->l);to->ab
=tp;tq=(to->ac+(to->ab-to->u))%tn->ah;tr=sc(tn,to,tq);sf(tn,tr,to->l);}static//
void si(fz const*tn,gk*to,int tp){int tq,tr,ts;ge*tt=tn->u.l;float const*tu=tn
->u.u;assert(tn->u.ar);tr=to->ad;ts=to[tp-1].ae;tt+=tr;tu+=tr*tn->u.ak;to->ac=0
;to->u=tt->l;to->ab=to->u-1;for(tq=tr;tq<ts;tq++){int tv,tw;tv=tt->l;tw=tt->u;
assert(tv>=to->u);while(tw>to->ab){assert((to->ab-to->u+1)<=tn->ah);if((to->ab-
to->u+1)==tn->ah){to->u++;to->ac++;}if(tn->bc){float*tx=se(tn,to,++to->ab);mv(
tn,to->ab,tx);}else{sh(tn,to,to->ab+1);}}sg(tn,to,tq,tv,tw,tu);++tt;tu+=tn->u.
ak;}}static void sj(fz const*tn,gk*to){float*tp=sc(tn,to,to->ac);sb(tn,((char*)
tn->ac)+((size_t)to->u*(size_t)tn->ae),tp,to->u);tp[0]=3.0e+38F;to->u++;if(++to
->ac==tn->ah)to->ac=0;}static void sk(fz const*tn,gk*to){float*tp=sc(tn,to,to->
ac);sf(tn,to->aj,tp);sb(tn,((char*)tn->ac)+((size_t)to->u*(size_t)tn->ae),to->
aj,to->u);tp[0]=3.0e+38F;to->u++;if(++to->ac==tn->ah)to->ac=0;}static void sl(
fz const*tn,gk*to,int tp,int tq,float const*tr,float const*ts,float const*tt){
assert(!tn->u.ar);;{int tu=0,tv=tq-tp+1;assert(tv>0);do{float*tw[8];int tx,ty=
tv;if(ty>8)ty=8;for(tx=0;tx<ty;tx++){tw[tx]=se(tn,to,tu+tx+tp);if((tx)&&(((tw[
tx])[0]==3.0e+38F)!=((tw[0])[0]==3.0e+38F))){ty=tx;break;}}((((tw[0])[0]==/////
3.0e+38F))?rz:sa)[ty-1](tw,tr+tu,ts,tt);tu+=ty;tv-=ty;}while(tv);};}typedef////
void sm(fz const*stbir_info,gk*split_info);static void sn(fz const*tn,gk*to,int
tp){int tq,tr,ts,tt,tu;ge*tv=tn->u.l;float const*tw=tn->u.u;sm*tx;void*ty;void*
tz;int ua,ub;int uc=(tn->bc)?(tn->an.l.u-tn->an.l.l+1):tn->l.ad.u;int ud=tn->be
*uc;assert(!tn->u.ar);tr=to->ad;ts=to[tp-1].ae;tt=to->ag;tu=to[tp-1].ah;tq=tt+
tn->u.am;tv+=tq;tw+=tn->u.ak*tq;if(tn->bc){tx=sk;ty=to->l;tz=((char*)ty)+sizeof
(float)*tn->be*(tn->an.l.u-tn->an.l.l+1);}else{tx=sj;ty=to->aj;tz=((char*)ty)+
sizeof(float)*tn->be*tn->l.ad.u;}to->u=tr;to->ab=-1;to->ac=-1;for(tq=0;tq<tn->
ah;tq++){float*ue=sc(tn,to,tq);ue[ud]=0.0f;ue[ud+1]=0.0f;ue[0]=3.0e+38F;}ua=1;
ub=tt;for(tq=tt;tq<tu;tq++){int uf,ug;uf=tv->l;ug=tv->u;assert(ug-uf+1<=tn->ah)
;if((ug>=uf)&&(((uf>=tr)&&(uf<ts))||((ug>=tr)&&(ug<ts)))){float const*uh=tw;ub=
tq;if((ua)&&(tq>tt))to->ag=tq;ua=0;if(uf<tr){uh+=tr-uf;uf=tr;}if(ug>=ts)ug=ts-1
;if(to->ac<0)to->ac=uf-tr;assert(to->ac<=uf);mv(tn,tq,to->l);if(!tn->bc)sf(tn,
to->aj,to->l);if(((to->ab-to->u+1)==tn->ah)&&(ug>to->ab))tx(tn,to);sl(tn,to,uf,
ug,uh,(float*)ty,(float*)tz);if(ug>to->ab)to->ab=ug;}++tv;tw+=tn->u.ak;}while(
to->u<ts)tx(tn,to);++ub;for(tq=0;tq<tp;tq++)if(to[tq].ah>ub)to[tq].ah=ub;}/////
static fx*so[]={0,hz,ic,ig,ih,ii,ie};static fy*sp[]={0,ib,ik,il,il,il,ij};/////
static void sq(gi*tn,ft to,fx*tp,fy*tq,fs tr,gh*ts,int tt,void*tu){if(to==0){to
=STBIR_FILTER_MITCHELL;if(ts->ab>=(1.0f-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<
20)/(1<<20)/(1<<20)))){if((ts->ab<=(1.0f+((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<
20)/(1<<20)/(1<<20))))&&(hd(ts->ad)==ts->ad))to=STBIR_FILTER_POINT_SAMPLE;else
to=STBIR_FILTER_CATMULLROM;}}tn->ag=to;assert(tn->ag!=0);assert((unsigned)tn->
ag<STBIR_FILTER_OTHER);tn->ah=so[to];tn->ai=sp[to];if(tp&&tq){tn->ah=tp;tn->ai=
tq;tn->ag=STBIR_FILTER_OTHER;}tn->aj=tr;tn->al=im(tn->ai,ts->ab,tu);tn->ar=0;if
(ts->ab>=(1.0f-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20))))tn->
ar=1;else if((tt)||(tn->al<=32))tn->ar=2;tn->ak=io(tn,tn->ar,tu);if(tr==///////
STBIR_EDGE_WRAP)if(tn->al>(ts->l*3))tn->al=ts->l*3;tn->am=tn->al/2;if(tr==/////
STBIR_EDGE_WRAP)if(tn->am>ts->l)tn->am=ts->l;tn->an=ip(tn,tn->ar);tn->ao=tn->an
*sizeof(ge);tn->ap=tn->an*tn->ak*sizeof(float)+sizeof(float)*3;tn->ab=0;tn->ac=
0;if(tn->ar==0){tn->at=tn->al;tn->as=ip(tn,2);tn->au=tn->as*sizeof(ge);tn->av=
tn->as*tn->at*sizeof(float);}}static void sr(gi*tn,ge*to,void*tp){float tq=tn->
ad.ab;float tr=tn->ad.ad;fy*ts=tn->ai;int tt=tn->ad.l;fs tu=tn->aj;float tv=tn
->ad.ac;assert(tn->ar!=0);if(tn->ar==1){int tw,tx;float ty=ts(tv,tp)*tq;iy(&tw,
&tx,0.5,ty,tv,tr,tt,tu);to->l=tw;iy(&tw,&tx,((float)(tn->ad.u-1))+0.5f,ty,tv,tr
,tt,tu);to->u=tx;}else if(tn->ar==2){float tz=ts(tq,tp)*tv;int ua=tn->am;int ub
=tn->ad.u;int uc;int ud;int ue,uf;iy(&ue,&uf,0,0,tv,tr,tt,tu);to->l=ue;iy(&ue,&
uf,(float)ub,0,tv,tr,tt,tu);to->u=uf;ud=to->l+1;uc=-ua;while(ud>=uc){int ug,uh;
jb(&ug,&uh,((float)ud)+0.5f,tz,tq,tr,ub);if(ug>uh)break;if((ug<ub)||(uh>=0))to
->l=ud;--ud;}ud=to->u-1;uc=ud+1+ua;while(ud<=uc){int ui,uj;jb(&ui,&uj,((float)
ud)+0.5f,tz,tq,tr,ub);if(ui>uj)break;if((ui<ub)||(uj>=0))to->u=ud;++ud;}}if(tn
->aj==STBIR_EDGE_WRAP){if((to->l>0)&&(to->u>=tt)){int uk=to->u-tt+1;if((uk+16)
>=to->l)to->l=0;}if((to->l<0)&&(to->u<(tt-1))){int ul=-to->l;if((tt-ul-16-1)<=
to->u)to->u=tt-1;}}else{if(to->l<0)to->l=0;if(to->u>=tt)to->u=tt-1;}}static////
void ss(gk*tn,int to,int tp,int tq,int tr,int ts,ge*tt){int tu,tv;int tw=tp;tv=
0;for(tu=0;tu<to;tu++){int tx;tn[tu].ad=tv;tx=tw/(to-tu);tn[tu].ae=tv+tx;if((ts
)&&(tu)){ge*ty;int tz,ua,ub,uc;ge*ud=tt+tv;ub=tq*3;if(tx<ub)ub=tx;ua=0;ty=ud;uc
=ty->l;for(tz=1;tz<=ub;tz++){++ud;if(ud->l>uc)break;if(ud->l<ty->l){ty=ud;ua=tz
;}}tn[tu-1].ae+=ua;tn[tu].ad+=ua;}tv+=tx;tw-=tx;tn[tu].ag=-tq;tn[tu].ah=tr+tq;}
}static void st(fz*tn){if(tn){{if(tn->ao){void*to=(tn->ao);(tn->ao)=0;((void)(
tn->al),free(to));}};}}static int su(int tn,int to){int tp;int tq=0;for(tp=0;tp
<tn;tp++){int tr=to/(tn-tp);if(tr>tq)tq=tr;to-=tr;}return tq;}static gn**sv[8]=
{0,nn,og,oz,ps,0,0,qk};static gn**sw[8]={0,no,oh,pa,pt,0,0,ql};static float sx[
5][8][4]={{{1.00000f,1.00000f,0.31250f,1.00000f},{0.56250f,0.59375f,0.00000f,//
0.96875f},{1.00000f,0.06250f,0.00000f,1.00000f},{0.00000f,0.09375f,1.00000f,///
1.00000f},{1.00000f,1.00000f,0.31250f,1.00000f},{0.03125f,0.12500f,1.00000f,///
1.00000f},{1.00000f,1.00000f,0.06250f,1.00000f},{0.00000f,1.00000f,0.00000f,///
0.03125f},},{{0.00000f,0.84375f,0.00000f,0.03125f},{0.09375f,0.93750f,0.00000f,
0.78125f},{0.87500f,0.21875f,0.00000f,0.96875f},{0.09375f,0.09375f,1.00000f,///
1.00000f},{0.00000f,0.84375f,0.00000f,0.03125f},{0.03125f,0.12500f,1.00000f,///
1.00000f},{1.00000f,1.00000f,0.06250f,1.00000f},{0.00000f,1.00000f,0.00000f,///
0.53125f},},{{0.00000f,0.53125f,0.00000f,0.03125f},{0.06250f,0.96875f,0.00000f,
0.53125f},{0.87500f,0.18750f,0.00000f,0.93750f},{0.00000f,0.09375f,1.00000f,///
1.00000f},{0.00000f,0.53125f,0.00000f,0.03125f},{0.03125f,0.12500f,1.00000f,///
1.00000f},{1.00000f,1.00000f,0.06250f,1.00000f},{0.00000f,1.00000f,0.00000f,///
0.56250f},},{{0.00000f,0.50000f,0.00000f,0.71875f},{0.06250f,0.84375f,0.00000f,
0.87500f},{1.00000f,0.50000f,0.50000f,0.96875f},{1.00000f,0.09375f,0.31250f,///
0.50000f},{0.00000f,0.50000f,0.00000f,0.71875f},{1.00000f,0.03125f,0.03125f,///
0.53125f},{1.00000f,1.00000f,0.06250f,1.00000f},{0.00000f,1.00000f,0.03125f,///
0.18750f},},{{0.00000f,0.59375f,0.00000f,0.96875f},{0.06250f,0.81250f,0.06250f,
0.59375f},{0.75000f,0.43750f,0.12500f,0.96875f},{0.87500f,0.06250f,0.18750f,///
0.43750f},{0.00000f,0.59375f,0.00000f,0.96875f},{0.15625f,0.12500f,1.00000f,///
1.00000f},{1.00000f,1.00000f,0.06250f,1.00000f},{0.00000f,1.00000f,0.03125f,///
0.34375f},}};typedef struct STBIR__V_FIRST_INFO{double l,u;int ab;int ac;int ad
;int ae;}sy;static int sz(float tn[8][4],int to,float tp,int tq,int tr,float ts
,int tt,int tu,sy*tv){double tw,tx;float*ty;int tz;int ua;if((tt<=4)||(tq<=4))
ua=(tt<tq)?6:7;else if((!tu)&&((tt<=16)||(tq<=16)))ua=4;else if(ts<=1.0f)ua=(tu
)?1:0;else if(ts<=2.0f)ua=2;else if(ts<=3.0f)ua=3;else ua=5;ty=tn[ua];tx=(float
)to*ty[0]+tp*(float)tr*ty[1];tw=(float)tr*ty[2]+ts*(float)to*ty[3];tz=(tw<=tx)?
1:0;if(tv){tv->u=tx;tv->l=tw;tv->ad=ua;tv->ac=tz;tv->ae=tu;}if((tv)&&(tv->ab))
tz=(tv->ab==2)?1:0;return tz;}static unsigned char ta[]={1,2,3,3,4,4,4,4,4,2,2,
4,4,4,4,2,2,};static gb tb[]={STBIRI_BGR,STBIRI_1CHANNEL,STBIRI_2CHANNEL,//////
STBIRI_RGB,STBIRI_RGBA,STBIRI_4CHANNEL,STBIRI_BGRA,STBIRI_ARGB,STBIRI_ABGR,////
STBIRI_RA,STBIRI_AR,STBIRI_RGBA_PM,STBIRI_BGRA_PM,STBIRI_ARGB_PM,STBIRI_ABGR_PM
,STBIRI_RA_PM,STBIRI_AR_PM,};static fz*td(gi*tn,gi*to,ge*tp,fr tq,fr tr,int ts,
int tt,int tu,int tv,void*tw){static char tx[8]={9,0,1,2,3,9,9,4};fz*ty=0;void*
tz=0;size_t ua=0;int ub;size_t uc,ud,ue,uf;int ug;int uh=0;int ui=su(ts,to->ad.
u);gb uj=tb[tq];gb uk=tb[tr];int ul=ta[uj];int um=ul;if((tn->ag!=//////////////
STBIR_FILTER_POINT_SAMPLE)||(to->ag!=STBIR_FILTER_POINT_SAMPLE)){if((uj>=//////
STBIRI_RGBA)&&(uj<=STBIRI_AR)&&(uk>=STBIRI_RGBA)&&(uk<=STBIRI_AR)){if(tv){uh=4;
}else{static int un[6]={7,7,7,7,3,3};uh=2;um=un[uj-STBIRI_RGBA];}}else if((uj>=
STBIRI_RGBA_PM)&&(uj<=STBIRI_AR_PM)&&(uk>=STBIRI_RGBA)&&(uk<=STBIRI_AR)){uh=3;}
else if((uj>=STBIRI_RGBA)&&(uj<=STBIRI_AR)&&(uk>=STBIRI_RGBA_PM)&&(uk<=////////
STBIRI_AR_PM)){uh=1;}}if(ul!=ta[uk])return 0;ub=sz(sx[(int)tx[um]],tn->al,tn->
ad.ab,tn->ad.u,to->al,to->ad.ab,to->ad.u,to->ar,0);uc=(tp->u-tp->l+1)*um*sizeof
(float)+sizeof(float)*3;ud=(size_t)tn->ad.u*(size_t)um*sizeof(float)+sizeof(///
float)*3;if(ub)ud=(uc+15)&~15;if((ud&4095)==0)ud+=64*3;ug=to->al+1;if((!to->ar)
&&(ug>ui))ug=ui;ue=(size_t)ug*(size_t)ud;uf=(size_t)tn->ad.u*(size_t)um*sizeof(
float)+sizeof(float);for(;;){int uo;void*up=tz;int uq=0;gi*ur=0;up=(void*)((((
size_t)up)+15)&~15);if(tz)ty=(fz*)up;up=(char*)(((size_t)up)+(sizeof(fz)));;up=
(void*)((((size_t)up)+15)&~15);if(tz)ty->ap=(gk*)up;up=(char*)(((size_t)up)+(//
sizeof(gk)*ts));;if(ty){static gm*us[6]={mm,mm,mm,mm,mn,mn};static go*ut[6]={mo
,mo,mo,mo,mp,mp};static gm*uu[6]={mq,mq,mq,mq,mr,mr};static go*uv[6]={ms,ms,ms,
ms,mt,mt};ty->ao=tz;ty->bg=ua;ty->bd=ul;ty->be=um;ty->ba=tt;ty->bb=tu;ty->av=(
int)ug;ty->ah=0;ty->ag=(int)ud;ty->aw=ts;ty->bc=ub;ty->ax=uj;ty->ay=uk;ty->ar=0
;ty->at=0;if(uh==2){ty->ar=us[uj-STBIRI_RGBA];ty->at=ut[uk-STBIRI_RGBA];}else//
if(uh==4){ty->ar=uu[uj-STBIRI_RGBA];ty->at=uv[uk-STBIRI_RGBA];}else if(uh==1){
ty->ar=uu[uj-STBIRI_RGBA];}else if(uh==3){ty->at=uv[uk-STBIRI_RGBA];}if(((uj==
STBIRI_RGB)&&(uk==STBIRI_BGR))||((uj==STBIRI_BGR)&&(uk==STBIRI_RGB))){if(tn->ad
.ab<1.0f)ty->at=mu;else ty->ar=mu;}}for(uo=0;uo<ts;uo++){up=(void*)((((size_t)
up)+15)&~15);if(tz)ty->ap[uo].l=(float*)up;up=(char*)(((size_t)up)+(uc));;up=(
void*)((((size_t)up)+15)&~15);if(tz)ty->ap[uo].ai=(float*)up;up=(char*)(((/////
size_t)up)+(ue));;up=(void*)((((size_t)up)+15)&~15);if(tz)ty->ap[uo].aj=(float*
)up;up=(char*)(((size_t)up)+(uf));;}if(to->ar==0){size_t uw;size_t ux;uw=(/////
size_t)to->au+(size_t)to->av;ux=(size_t)(uc+ue+uf)*(size_t)ts;if(ux>=uw){if(ty)
{to->ab=(ge*)ty->ap[0].l;to->ac=(float*)(((char*)ty->ap[0].l)+to->au);}}else{up
=(void*)((((size_t)up)+15)&~15);if(tz)to->ab=(ge*)up;up=(char*)(((size_t)up)+(
to->au));;up=(void*)((((size_t)up)+15)&~15);if(tz)to->ac=(float*)up;up=(char*)(
((size_t)up)+(to->av));;}}up=(void*)((((size_t)up)+15)&~15);if(tz)tn->l=(ge*)up
;up=(char*)(((size_t)up)+(tn->ao));;up=(void*)((((size_t)up)+15)&~15);if(tz)tn
->u=(float*)up;up=(char*)(((size_t)up)+(tn->ap));;if((tn->ah==to->ah)&&(tn->ai
==to->ai)&&(tn->aj==to->aj)&&(tn->ad.u==to->ad.u)){float uy=tn->ad.ab-to->ad.ab
;float uz=tn->ad.ad-to->ad.ad;if(uy<0.0f)uy=-uy;if(uz<0.0f)uz=-uz;if((uy<=((///
float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))&&(uz<=((float)1/(1<<
20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))){if(tn->ar==to->ar){uq=1;goto////
no_vert_alloc;}ur=tn;}}up=(void*)((((size_t)up)+15)&~15);if(tz)to->l=(ge*)up;up
=(char*)(((size_t)up)+(to->ao));;up=(void*)((((size_t)up)+15)&~15);if(tz)to->u=
(float*)up;up=(char*)(((size_t)up)+(to->ap));;no_vert_alloc:if(ty){;jf(tn,0,tw)
;ty->as=sv[um][tn->aq.ab&3];if(tn->aq.ab<=12)ty->as=sw[um][tn->aq.ab-1];ty->an.
l.l=tp->l;ty->an.l.u=tp->u;ix(tn,&ty->an);tn->ak=je(tn->an,tn->l,tn->u,tn->ak,
tn->aq.ab,ty->an.l.l,ty->an.l.u);hx(&ty->l,tn,sizeof(gi));;if(uq){hx(&ty->u,tn,
sizeof(gi));}else{;jf(to,ur,tw);hx(&ty->u,to,sizeof(gi));;}ss(ty->ap,ty->aw,ty
->u.ad.u,ty->u.am,ty->u.ad.l,ty->u.ar,ty->u.l);ty->ah=ty->u.aq.ab;if((!ty->u.ar
)&&(ty->ah>ui))ty->ah=ui;assert(ty->ah<=ty->av);}if(ty==0){ua=(15+(size_t)up);
tz=((void)(tw),malloc(ua));if(tz==0)return 0;}else return ty;}}static int te(fz
const*tn,int to,int tp){gk*tq=tn->ap+to;;;if(tn->u.ar)si(tn,tq,tp);else sn(tn,
tq,tp);;return 1;}static void tf(fz*tn,ga*to){static gl*tp[////////////////////
STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]={jk,jk,0,jx,jv,};static gl*tq[//
STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]={{jm,jk
,0,jx,jv},{kf,kd,0,kn,kl},{kv,kt,0,ld,lb},{ll,lj,0,lu,ls},{jp,jk,0,jx,jv},{mc,
ma,0,mk,mi},};static gl*tr[2][2]={{jg,ji},{jr,jt},};static gl*ts[STBIRI_AR-////
STBIRI_RGBA+1][2][2]={{{jg,ji},{jr,jt}},{{jz,kb},{kh,kj}},{{kp,kr},{kx,kz}},{{
lf,lh},{ln,lq}},{{jg,ji},{jr,jt}},{{lw,ly},{me,mg}}};static gp*tt[/////////////
STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]={jl,jl,0,jy,jw,};static gp*tu[//
STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]={{jo,jl
,0,jy,jw},{kg,ke,0,ko,km},{kw,ku,0,le,lc},{lm,lk,0,lv,lt},{jq,jl,0,jy,jw},{md,
mb,0,ml,mj}};static gp*tv[2][2]={{jh,jj},{js,ju},};static gp*tw[STBIRI_AR-/////
STBIRI_RGBA+1][2][2]={{{jh,jj},{js,ju}},{{ka,kc},{ki,kk}},{{kq,ks},{ky,la}},{{
lg,li},{lp,lr}},{{jh,jj},{js,ju}},{{lx,lz},{mf,mh}}};gl*tx=0;gp*ty=0;fu tz,ua;
tz=to->az;ua=to->ba;tn->ab=to->u;tn->ad=to->ar;tn->ae=to->as;if((tn->l.ag==////
STBIR_FILTER_POINT_SAMPLE)&&(tn->u.ag==STBIR_FILTER_POINT_SAMPLE)){if(((tz==///
STBIR_TYPE_UINT8_SRGB)||(tz==STBIR_TYPE_UINT8_SRGB_ALPHA))&&((ua==/////////////
STBIR_TYPE_UINT8_SRGB)||(ua==STBIR_TYPE_UINT8_SRGB_ALPHA))){tz=STBIR_TYPE_UINT8
;ua=STBIR_TYPE_UINT8;}}if(tn->ad==0)tn->ad=tn->bd*tn->l.ad.l*gd[tz];if(tn->ae==
0)tn->ae=tn->bd*tn->l.ad.u*gd[ua];tn->ac=((char*)to->aj)+((size_t)tn->bb*(/////
size_t)to->as)+(tn->ba*tn->bd*gd[ua]);tn->ak=to->ai;tn->al=to->l;tn->am=to->aq;
if((tz==STBIR_TYPE_UINT8)||(tz==STBIR_TYPE_UINT16)){int ub=0;if((!tn->ar)&&(!tn
->at))if(((tz==STBIR_TYPE_UINT8)&&(ua==STBIR_TYPE_UINT8))||((tz==//////////////
STBIR_TYPE_UINT16)&&(ua==STBIR_TYPE_UINT16)))ub=1;if(tn->ax<=STBIRI_4CHANNEL)tx
=tr[tz==STBIR_TYPE_UINT16][ub];else tx=ts[(tn->ax-STBIRI_RGBA)%(STBIRI_AR-/////
STBIRI_RGBA+1)][tz==STBIR_TYPE_UINT16][ub];}else{if(tn->ax<=STBIRI_4CHANNEL)tx=
tp[tz-STBIR_TYPE_UINT8_SRGB];else tx=tq[(tn->ax-STBIRI_RGBA)%(STBIRI_AR-///////
STBIRI_RGBA+1)][tz-STBIR_TYPE_UINT8_SRGB];}if((ua==STBIR_TYPE_UINT8)||(ua==////
STBIR_TYPE_UINT16)){int uc=0;if((!tn->ar)&&(!tn->at))if(((tz==STBIR_TYPE_UINT8)
&&(ua==STBIR_TYPE_UINT8))||((tz==STBIR_TYPE_UINT16)&&(ua==STBIR_TYPE_UINT16)))
uc=1;if(tn->ay<=STBIRI_4CHANNEL)ty=tv[ua==STBIR_TYPE_UINT16][uc];else ty=tw[(tn
->ay-STBIRI_RGBA)%(STBIRI_AR-STBIRI_RGBA+1)][ua==STBIR_TYPE_UINT16][uc];}else{
if(tn->ay<=STBIRI_4CHANNEL)ty=tt[ua-STBIR_TYPE_UINT8_SRGB];else ty=tu[(tn->ay-
STBIRI_RGBA)%(STBIRI_AR-STBIRI_RGBA+1)][ua-STBIR_TYPE_UINT8_SRGB];}tn->ai=tz;tn
->aj=ua;tn->aq=tx;tn->au=ty;}static void tg(int*tn,int*to,int tp,double*tq,////
double*tr){double ts,tt;int tu;if(*tn<0){ts=((double)*tn)/((double)*to);tt=ts*(
*tr-*tq);*tq-=tt;*tn=0;}tu=tp-(*tn+*to);if(tu<0){ts=((double)tu)/((double)*to);
tt=ts*(*tr-*tq);*tr+=tt;*to=tp-*tn;}}static int ti(double tn,fp to,fp*tp,fp*tq,
int tr){double ts;fq tt,tu;fq tv=0;fq tw=1;fq tx=1;fq ty=0;tt=(fq)(tn*(double)(
1<<25));tu=1<<25;for(;;){fq tz,ua;if(((tr)?ty:tx)>=to)break;if(ty){ts=((double)
tx/(double)ty)-tn;if(ts<0.0)ts=-ts;if(ts<(1.0/(double)(1<<24))){*tp=(fp)tx;*tq=
(fp)ty;return 1;}}if(tu==0)break;tz=tt/tu;ua=tt%tu;tt=tu;tu=ua;ua=tz*ty+tw;tw=
ty;ty=ua;ua=tz*tx+tv;tv=tx;tx=ua;}if(tr){tx=(fq)(tn*(double)to+0.5);ty=to;}else
{tx=to;ty=(fq)(((double)to/tn)+0.5);}*tp=(fp)tx;*tq=(fp)ty;ts=(ty)?(((double)(
fp)tx/(double)(fp)ty)-tn):1.0;if(ts<0.0)ts=-ts;return(ts<(1.0/(double)(1<<24)))
?1:0;}static int tj(gh*tn,int to,int*tp,int tq,int tr,double ts,double tt){////
double tu,tv,tw,tx,ty,tz;tx=tt-ts;if((to==0)||(tr==0)||(tq==0)||(tx<=((float)1/
(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20))))return 0;if((*tp>=to)||((*tp+
tq)<=0)||(ts>=(1.0f-((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))
)||(tt<=((float)1/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20))))return 0;tu
=(double)to;tv=(double)tr;tw=((double)tq)/tu;ty=tw/tx;tz=(tu/tv)*ty;tn->ab=(///
float)tz;tn->ac=(float)(1.0/tz);tg(tp,&tq,to,&ts,&tt);tx=tt-ts;if(tx<=((float)1
/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)/(1<<20)))return 0;tn->ad=(float)(ts*ty
*tu);tn->ae=ti(tz,(tz<=1.0)?to:tr,&tn->ag,&tn->ah,(tz>=1.0));tn->l=tr;tn->u=tq;
return 1;}static void tk(ga*tn,fr to,fu tp){tn->ai=0;tn->aq=0;tn->l=tn;tn->bk=0
;tn->aw=0;tn->bb=STBIR_FILTER_DEFAULT;tn->bg=0;tn->bh=0;tn->bc=////////////////
STBIR_FILTER_DEFAULT;tn->bi=0;tn->bj=0;tn->bd=STBIR_EDGE_CLAMP;tn->be=/////////
STBIR_EDGE_CLAMP;tn->ad=0;tn->ae=0;tn->ag=1;tn->ah=1;tn->am=0;tn->an=0;tn->ao=
tn->ak;tn->ap=tn->al;tn->az=tp;tn->ba=tp;tn->ax=to;tn->ay=to;tn->av=1;}extern//
void stbir_resize_init(ga*ty,const void*tz,int ua,int ub,int uc,void*ud,int ue,
int uf,int ug,fr uh,fu ui){ty->u=tz;ty->ab=ua;ty->ac=ub;ty->ar=uc;ty->aj=ud;ty
->ak=ue;ty->al=uf;ty->as=ug;ty->au=0;tk(ty,uh,ui);}static int tl(ga*tn,int to){
ge tp={0,0};gi tq,tr;int ts,tt;fz*tu;if(tn->bk)return 0;assert(!((unsigned)tn->
bb>=STBIR_FILTER_OTHER));if((unsigned)tn->bb>=STBIR_FILTER_OTHER)return 0;/////
assert(!((unsigned)tn->bc>=STBIR_FILTER_OTHER));if((unsigned)tn->bc>=//////////
STBIR_FILTER_OTHER)return 0;if(to<=0)return 0;;ts=tn->am;tt=tn->an;if(!tj(&tq.
ad,tn->ak,&ts,tn->ao,tn->ab,tn->ad,tn->ag))return 0;if(!tj(&tr.ad,tn->al,&tt,tn
->ap,tn->ac,tn->ae,tn->ah))return 0;if((tq.ad.u==0)||(tr.ad.u==0))return 0;sq(&
tq,tn->bb,tn->bg,tn->bh,tn->bd,&tq.ad,1,tn->l);sr(&tq,&tp,tn->l);sq(&tr,tn->bc,
tn->bi,tn->bj,tn->be,&tr.ad,0,tn->l);if((tr.ad.u/to)<4){to=tr.ad.u/4;if(to==0)
to=1;};tu=td(&tq,&tr,&tp,tn->ax,tn->ay,to,ts,tt,tn->au,tn->l);;;if(tu){tn->at=
to;tn->bk=tu;tn->av=0;tf(tu,tn);return to;}return 0;}extern void///////////////
stbir_free_samplers(ga*to){if(to->bk){st(to->bk);to->bk=0;to->aw=0;}}extern int
stbir_build_samplers_with_splits(ga*tp,int tq){if((tp->bk==0)||(tp->av)){if(tp
->bk)stbir_free_samplers(tp);tp->aw=1;return tl(tp,tq);};return 1;}extern int//
stbir_build_samplers(ga*to){return stbir_build_samplers_with_splits(to,1);}////
extern int stbir_resize_extended(ga*to){int tp;if((to->bk==0)||(to->av)){int tq
=to->aw;if(to->bk){st(to->bk);to->bk=0;}if(!stbir_build_samplers(to))return 0;
to->aw=tq;if(to->bk==0)return 1;}else{;}tp=te(to->bk,0,to->at);if(!to->aw){////
stbir_free_samplers(to);to->bk=0;}return tp;}static void*tm(const void*tn,int//
to,int tp,int tq,void*tr,int ts,int tt,int tu,fr tv,fu tw,fs tx,ft ty){ga tz;//
int ua;int ub;void*uc;void*ud;ua=ts*gd[tw]*ta[tb[tv]];if(ua==0)return 0;if(tu==
0)tu=ua;ub=tu;if(ub<0)ub=-ub;if(ub<ua)return 0;uc=tr;ud=0;if(tr==0){size_t ue;
char*uf;ue=(size_t)ub*(size_t)tt;if(ue==0)return 0;uf=(char*)((void)(0),malloc(
ue));if(uf==0)return 0;ud=uf;if(tu<0)uc=uf+((size_t)ub*(size_t)(tt-1));else uc=
uf;}stbir_resize_init(&tz,tn,to,tp,tq,uc,ts,tt,tu,tv,tw);tz.bd=tx;tz.be=tx;tz.
bb=ty;tz.bc=ty;if(!stbir_resize_extended(&tz)){if(ud)((void)(0),free(ud));/////
return 0;}return(ud)?ud:uc;}extern void*stbir_resize(const void*tz,int ua,int//
ub,int uc,void*ud,int ue,int uf,int ug,fr uh,fu ui,fs uj,ft uk){return(void*)tm
(tz,ua,ub,uc,ud,ue,uf,ug,uh,ui,uj,uk);}

// clang-format on
// NOLINTEND

// The actual code starts here :D

#ifdef _WIN32
#  include <time.h>
#  include <windows.h>
#else
#  undef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 199309L
#  include <time.h>
#endif

// Default values for cli
#define DEFAULT_SEQLEN      16384
#define DEFAULT_TOPK        0
#define DEFAULT_CHUNK_SIZE  1024
#define DEFAULT_TEMPERATURE 1.0
#define DEFAULT_TOPP        1.0
#define DEFAULT_RPEN        1.0
#define DEFAULT_PROMPT      "Once upon a time"

#define TOSTRING(x) STRINGIFY_(x)
// I usually use a trailing underscore to indicate a variable/function/macro is
// temporary
#define STRINGIFY_(x) #x

// Matrix multiplication block sizes
#define GEMM_NT_MR 8
#define GEMM_NT_NR 8
#define GEMM_NN_MR 8
#define GEMM_I8_MR 8
#define GEMM_I8_NR 8

#define PREFILL_FALLBACK_THRESHOLD 8

// Block size of blockwise causal masking
#define QK_BLOCK_SIZE 64

// To compile the code for a different dtype, simply append -DDTYPE=... (e.g.
// -DTYPE=DTYPE_BF16)
#define DTYPE_FP16 1
#define DTYPE_BF16 2
#define DTYPE_FP32 3

#ifndef DTYPE
#  define DTYPE DTYPE_FP16
#endif

// clang-format off
#if DTYPE == DTYPE_FP16
#  define floatx     _Float16
#  define DTYPE_STR  "float16"
#  define DTYPE_CODE 1
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007BFF}).f)
#elif DTYPE == DTYPE_BF16
#  define floatx     __bf16
#  define DTYPE_STR  "bfloat16"
#  define DTYPE_CODE 2
#  define FLOATX_MAX (((union {floatx f; uint16_t b; }){.b = 0x00007F7F}).f)
#elif DTYPE == DTYPE_FP32
#  define floatx     float
#  define DTYPE_STR  "float32"
#  define DTYPE_CODE 3
#  define FLOATX_MAX (((union {floatx f; uint32_t b; }){.b = 0x7F7FFFFF}).f)
#else
#  error "unsupported DTYPE"
#endif

// clang-format on

// Global interruption flag
static volatile sig_atomic_t g_interrupted = 0;

#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

/* */
static inline int
max(int a, int b)
{
  return a > b ? a : b;
}

/* */
static inline int
min(int a, int b)
{
  return a < b ? a : b;
}

/* */
static void
signal_handler(int signum)
{
  (void)signum;
  g_interrupted = 1;
}

// Cross-platform interruption handling

#ifdef _WIN32
#  include <windows.h>
#  define SLEEP_SEC(sec) Sleep((sec) * 1000)
#  define GETPID()       GetCurrentProcessId()
/* */
BOOL WINAPI
console_handler_(DWORD dwCtrlType)
{
  switch (dwCtrlType)
  {
    case CTRL_C_EVENT:         // Ctrl+C
    case CTRL_BREAK_EVENT:     // Ctrl+Break
    case CTRL_CLOSE_EVENT:     // Closed console window
    case CTRL_LOGOFF_EVENT:    // User logoff
    case CTRL_SHUTDOWN_EVENT:  // System shutdown
      g_interrupted = 1;
      return TRUE;
    default:
      return FALSE;
  }
}
#else
#  include <unistd.h>
#  define SLEEP_SEC(sec) sleep(sec)
#  define GETPID()       getpid()
#endif

/* */
void
setup_signal_handler(void)
{
#ifdef _WIN32
  if (!SetConsoleCtrlHandler(console_handler_, TRUE))
  {
    // Should almost never happen
    signal(SIGINT, signal_handler);
  }
#else
  struct sigaction sa;
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;

  if (sigaction(SIGINT, &sa, NULL) == -1)
  {
    perror("sigaction(SIGINT)");
  }
  if (sigaction(SIGTERM, &sa, NULL) == -1)
  {
    perror("sigaction(SIGTERM)");
  }
#endif
}

// Cross-platform wall time

/* */
static double
now_sec(void)
{
#ifdef _WIN32
  static LARGE_INTEGER freq;
  static int           freq_init = 0;
  LARGE_INTEGER        counter;

  if (!freq_init)
  {
    QueryPerformanceFrequency(&freq);
    freq_init = 1;
  }
  QueryPerformanceCounter(&counter);
  return (double)counter.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Poor man's memory wrappers, I just really dislike the redundency of writing
 * ```c
 * if ((ptr = malloc(count * sizeof(*ptr))) == NULL)
 * {
 *   fprintf(stderr, "error: ...");
 *   goto fail;
 * }
 * ```
 * in every memory allocation. So I wrapped them into the following macros, now
 * I only need to type:
 * ```c
 * MALLOC(ptr, count, "name_of_the_buffer", goto fail;);
 * ```
 * (Same for other operations like calloc or fread)
 * This is much more readable and less redundent for me. The implementations of
 * the macros themselves are indeed error-prone and hard to read, but once
 * they're complete I barely need to touch them. I guess that worked for me :D
 */

#define MALLOC(ptr, count, name, fail)                                     \
  do                                                                       \
  {                                                                        \
    ptr = malloc((count) * sizeof(*(ptr)));                                \
    if (ptr == NULL)                                                       \
    {                                                                      \
      fprintf(stderr, "error: memory allocation failed: %s (size: %zu)\n", \
          (name), (size_t)(count) * sizeof(*(ptr)));                       \
      fail                                                                 \
    }                                                                      \
  }                                                                        \
  while (0)

#define CALLOC(ptr, count, name, fail)                                     \
  do                                                                       \
  {                                                                        \
    ptr = calloc((count), sizeof(*(ptr)));                                 \
    if (ptr == NULL)                                                       \
    {                                                                      \
      fprintf(stderr, "error: memory allocation failed: %s (size: %zu)\n", \
          (name), (size_t)(count) * sizeof(*(ptr)));                       \
      fail                                                                 \
    }                                                                      \
  }                                                                        \
  while (0)

#define FGETC(var, fp, name, fail)                            \
  do                                                          \
  {                                                           \
    var = fgetc(fp);                                          \
    if (var == EOF)                                           \
    {                                                         \
      fprintf(stderr, "error: File read failed: %s", (name)); \
      fail                                                    \
    }                                                         \
  }                                                           \
  while (0)

#define FREAD(ptr, count, fp, name, fail)                                      \
  do                                                                           \
  {                                                                            \
    size_t fread_buf_ = fread((ptr), sizeof(*(ptr)), (count), (fp));           \
    if (fread_buf_ != (size_t)(count))                                         \
    {                                                                          \
      fprintf(stderr, "error: file read failed: %s (expected %zu, got %zu)\n", \
          (name), (size_t)(count), fread_buf_);                                \
      fail                                                                     \
    }                                                                          \
  }                                                                            \
  while (0)

#define READ_UINT16(var, fp, name, fail)                \
  do                                                    \
  {                                                     \
    /* Big-endian */                                    \
    unsigned char ru16_buf_[2];                         \
    FREAD(ru16_buf_, 2, (fp), (name), fail);            \
    var = ((int)ru16_buf_[0] << 8) | (int)ru16_buf_[1]; \
  }                                                     \
  while (0)

#define READ_UINT32(var, fp, name, fail)                                    \
  do                                                                        \
  {                                                                         \
    /* Big-endian */                                                        \
    unsigned char ru32_buf_[4];                                             \
    FREAD(ru32_buf_, 4, (fp), (name), fail);                                \
    var = ((uint32_t)ru32_buf_[0] << 24) | ((uint32_t)ru32_buf_[1] << 16) | \
          ((uint32_t)ru32_buf_[2] << 8) | ((uint32_t)ru32_buf_[3]);         \
  }                                                                         \
  while (0)

#define READ_FP32(var, fp, name, fail)            \
  do                                              \
  {                                               \
    int rfp32_bits_;                              \
    READ_UINT32(rfp32_bits_, (fp), (name), fail); \
    memcpy(&var, &rfp32_bits_, sizeof(float));    \
  }                                               \
  while (0)

// Pascal-style string: first byte is length, then that many chars
#define READ_STR(var, fp, data, offset, name, fail)     \
  do                                                    \
  {                                                     \
    char rstr_len_name_[128];                           \
    snprintf(rstr_len_name_, 128, "%s.length", (name)); \
    int rstr_len_;                                      \
    FGETC(rstr_len_, fp, rstr_len_name_, fail);         \
    var = (data) + *(offset);                           \
    FREAD(var, rstr_len_, (fp), (name), fail);          \
    (data)[*(offset) + rstr_len_] = '\0';               \
    *(offset) += rstr_len_ + 1;                         \
  }                                                     \
  while (0)

#define READ_TENSOR(ptr, count, fp, name, fail) \
  do                                            \
  {                                             \
    MALLOC((ptr), (count), (name), fail);       \
    FREAD((ptr), (count), (fp), (name), fail);  \
  }                                             \
  while (0)

// Linear layer weights: either plain floatx or int8 + per-channel scales
#define READ_LINEAR(w, fp, m, n, quant, name, fail)                            \
  do                                                                           \
  {                                                                            \
    MALLOC((w), 1, (name), fail);                                              \
    if (!(quant))                                                              \
    {                                                                          \
      (w)->dtype = DTYPE_FPX;                                                  \
      char rlinear_fpx_name_[128];                                             \
      snprintf(rlinear_fpx_name_, 128, "%s.fpx", (name));                      \
      READ_TENSOR(                                                             \
          (w)->fpx, (size_t)(m) * (size_t)(n), (fp), rlinear_fpx_name_, fail); \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      (w)->dtype = DTYPE_INT8;                                                 \
      char rlinear_i8q_name_[128];                                             \
      char rlinear_i8scales_name_[128];                                        \
      snprintf(rlinear_i8q_name_, 128, "%s.i8.q", (name));                     \
      snprintf(rlinear_i8scales_name_, 128, "%s.i8.scales", (name));           \
      READ_TENSOR((w)->i8.q, (size_t)(m) * (size_t)(n), (fp),                  \
          rlinear_i8q_name_, fail);                                            \
      READ_TENSOR((w)->i8.scales, (n), (fp), rlinear_i8scales_name_, fail);    \
    }                                                                          \
  }                                                                            \
  while (0)

/* Peek at how many bytes a sequence of pascal strings will occupy */
static inline int
get_strarr_bytes(FILE *fp, int count)
{
  // Read a sequence of pascal-stype strings
  int  offset = 0;
  long pos    = ftell(fp);
  if (pos == -1L)
  {
    perror("ftell failed");
    return -1;
  }
  // Get the total number of bytes
  for (int i = 0; i < count; i++)
  {
    int len;
    FGETC(len, fp, "<str.length>", return -1;);
    if (fseek(fp, len, SEEK_CUR) != 0)
    {
      perror("fseek failed");
      return -1;
    }
    offset += len + 1;
  }
  // Resume position
  if (fseek(fp, pos, SEEK_SET) != 0)
  {
    perror("fseek failed");
    return -1;
  }
  return offset;
}

/* Configuration for the SigLIP vision encoder */
typedef struct
{
  int   n_layers;    // ViT layers
  int   image_size;  // Pixels per image side
  int   patch_size;  // Patches per image side
  int   hidden_dim;  // ViT dimension
  int   n_heads;     // Attention heads
  int   mlp_dim;     // Intermediate size in FFN
  float eps;         // LayerNorm epsilon
} VisionConfig;

/* Configuration for the main Gemma text encoder */
typedef struct
{
  int   n_layers;       // Transformer layers
  int   n_heads;        // Attention heads
  int   n_kv_heads;     // Key/value heads (GQA)
  int   head_dim;       // Dim per head
  int   embed_dim;      // Model dimension
  int   mlp_dim;        // Intermediate size in FFN
  int   q_scale;        // Scale applied to queries before attention
  int   slide_len;      // Sliding-window size
  int   image_toks;     // Number of soft tokens per image
  int   max_seqlen;     // Max number of position embeddings
  int   vocab_size;     // Number of possible tokens
  float local_theta;    // RoPE base for local (sliding) attention
  float global_theta;   // RoPE base for full attention
  float eps;            // RMSNorm epsilon
  float att_softcap;    // tanh softcap on attention scores (0 = off)
  float logit_softcap;  // tanh softcap on final logits (0 = off)
  bool *att_layers;     // true = use sliding window for that layer
  bool  support_mm;     // model has a vision tower
  bool  qk_norm;        // query/key RMSNorm
  bool  pre_mlp_norm;
  bool  pst_mlp_norm;
} TextConfig;

/* */
static void
free_text_config(TextConfig *cfg)
{
  if (cfg == NULL) return;
  free(cfg->att_layers);
  free(cfg);
}

// Tokenizer implementation

/* */
typedef struct
{
  char *val;
  int   idx;
} Token;
typedef struct
{
  char *str1;
  char *str2;
  int   rank;
} Merge;

/* */
static int
cmp_token(const void *a, const void *b)
{
  return strcmp(((Token *)a)->val, ((Token *)b)->val);
}

/* */
static int
cmp_merge(const void *a, const void *b)
{
  int ret = strcmp(((Merge *)a)->str1, ((Merge *)b)->str1);
  if (ret != 0)
  {
    return ret;
  }
  return strcmp(((Merge *)a)->str2, ((Merge *)b)->str2);
}

/* */
typedef struct
{
  int n_merges;
  int vocab_size;
  int bos;  // beginning of sequence
  int eos;  // end of sequence
  int sot;  // start of turn
  int eot;  // end of turn
  int soi;  // start of image
  int eoi;  // end of image
  int ist;  // image soft token

  char  *vocab_data;
  char  *merge_data;
  char **vocab;
  Token *vocab_sorted;  // sorted for binary search
  Merge *ranks;         // sorted merges for BPE
} GemmaTokenizer;

/* */
static void
free_tokenizer(GemmaTokenizer *tok)
{
  if (tok == NULL) return;
  free(tok->vocab_data);
  free(tok->merge_data);
  free(tok->vocab);
  free(tok->vocab_sorted);
  free(tok->ranks);
  free(tok);
}

/* Look up a string in the sorted vocab -> token id (or -1) */
static int
get_token_idx(GemmaTokenizer *tok, char *str)
{
  // Get idx_to_vocab[string]
  Token  key = {.val = str};
  Token *val = bsearch(&key, tok->vocab_sorted, tok->vocab_size,
      sizeof(tok->vocab_sorted[0]), cmp_token);
  if (val == NULL)
  {
    return -1;
  }
  return val->idx;
}

/* Look up a (str1, str2) pair in the merges table */
static Merge *
get_merge_rec(GemmaTokenizer *tok, char *str1, char *str2)
{
  // Get merges[(str1, str2)]
  Merge  key = {.str1 = str1, .str2 = str2};
  Merge *val = bsearch(
      &key, tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);
  return val;  // NULL if not found
}

/* Minimal byte-pair encoding algorithm implementation */
int *
encode(
    GemmaTokenizer *tok, const char *sstr, int len, int *tokens, int *n_tokens)
{
  unsigned char *str = (unsigned char *)sstr;

  // Convert UTF-8 string to tokens of individual codepoints
  int i     = 0;
  int tok_i = 0;

  while (i < len)
  {
    /* table from https://zh.wikipedia.org/wiki/UTF-8
     *
     * U+00000-U+00007F  1  0xxxxxxx
     * U+00080-U+0007FF  2  110xxxxx  10xxxxxx
     * U+00800-U+00FFFF  3  1110xxxx  10xxxxxx  10xxxxxx
     * U+10000-U+1FFFFF  4  11110xxx  10xxxxxx  10xxxxxx  10xxxxxx
     */
    int n_bytes = 0;
    int start   = i;

    if (str[i] >> 7 == 0)
    {
      n_bytes = 1;
    }
    else if (i + 1 < len && str[i] >> 5 == 6 && str[i + 1] >> 6 == 2)
    {
      n_bytes = 2;
    }
    else if (i + 2 < len && str[i] >> 4 == 14 && str[i + 1] >> 6 == 2 &&
             str[i + 2] >> 6 == 2)
    {
      n_bytes = 3;
    }
    else if (i + 3 < len && str[i] >> 3 == 30 && str[i + 1] >> 6 == 2 &&
             str[i + 2] >> 6 == 2 && str[i + 3] >> 6 == 2)
    {
      n_bytes = 4;
    }
    else
    {
      n_bytes = 1;
    }

    char cstr[5];
    for (int b = 0; b < n_bytes; b++)
    {
      cstr[b] = (char)str[i++];
    }
    cstr[n_bytes] = '\0';

    int token = get_token_idx(tok, cstr);
    if (token == -1)
    {
      // Fallback to byte level tokens
      char bstr[10];
      for (int b = start; b < i; b++)
      {
        snprintf(bstr, 10, "<0x%02X>", (unsigned char)str[b]);
        tokens[tok_i++] = get_token_idx(tok, bstr);
      }
    }
    else
    {
      tokens[tok_i++] = token;
    }
  }

  // Keep merging the highest-rank pair until nothing left
  for (;;)
  {
    int    best_rank = 2147483647;
    Merge *best_pair = NULL;

    // Find the merge with best rank
    for (int i = 0; i < tok_i - 1; i++)
    {
      char *str1 = tok->vocab[tokens[i]];
      char *str2 = tok->vocab[tokens[i + 1]];

      Merge *merge = get_merge_rec(tok, str1, str2);
      if (merge != NULL && merge->rank < best_rank)
      {
        best_rank = merge->rank;
        best_pair = merge;
      }
    }

    if (best_pair == NULL) break;  // No more merges

    int i = 0;
    while (i < tok_i - 1)
    {
      Merge pair = {
          .str1 = tok->vocab[tokens[i]], .str2 = tok->vocab[tokens[i + 1]]};
      if (cmp_merge(&pair, best_pair) == 0)
      {
        // Merge the pair, left shift all the tokens on its right side
        char merged[128];
        snprintf(
            merged, sizeof(merged), "%s%s", best_pair->str1, best_pair->str2);
        tokens[i] = get_token_idx(tok, merged);
        for (int j = i + 1; j < tok_i - 1; j++)
        {
          tokens[j] = tokens[j + 1];
        }
        tok_i--;
      }
      i++;
    }
  }

  *n_tokens += tok_i;
  return tokens;
}

/* */
const char *
decode(GemmaTokenizer *tok, int id, char *byte_buf)
{
  char *s = tok->vocab[id];
  // Byte-fallback tokens look like <0xAB>, turn them back into a raw byte
  size_t len = strlen(s);
  if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>')
  {
    unsigned int byte_val = 0;
    if (sscanf(s + 3, "%2x", &byte_val) == 1)
    {
      byte_buf[0] = (char)byte_val;
      byte_buf[1] = '\0';
      return byte_buf;
    }
  }
  return s;
}

// Weight storage

/* */
typedef enum
{
  DTYPE_FPX,
  DTYPE_INT8
} WeightDType;

/* A linear layer, either dense fpx or int8 + scales */
typedef struct
{
  WeightDType dtype;
  union
  {
    floatx *fpx;
    struct
    {
      int8_t *q;
      floatx *scales;
    } i8;
  };
} Linear;

/* */
static void
free_linear(Linear *l)
{
  if (l == NULL) return;
  if (l->dtype == DTYPE_FPX)
  {
    free(l->fpx);
  }
  else
  {
    free(l->i8.q);
    free(l->i8.scales);
  }
  free(l);
}

/* One Gemma transformer block */
typedef struct
{
  // Attention weights
  // 2D weights are stored transposed for better GEMV cache use

  Linear *wq;  // (embed_dim, n_heads * head_dim).T
  Linear *wk;  // (embed_dim, n_kv_heads * head_dim).T
  Linear *wv;  // (embed_dim, n_kv_heads * head_dim).T
  Linear *wo;  // (n_heads * head_dim, embed_dim).T

  // Feedforward weights
  Linear *w1;  // (embed_dim, mlp_dim).T
  Linear *w2;  // (embed_dim, mlp_dim).T
  Linear *w3;  // (mlp_dim, embed_dim).T

  // RMSNorm weights (Gemma adds 1.0 to the weight)
  floatx *nq;  // (head_dim,)
  floatx *nk;  // (head_dim,)
  floatx *n1;  // (embed_dim,)
  floatx *n2;  // (embed_dim,)
  floatx *n3;  // (embed_dim,)
  floatx *n4;  // (embed_dim,)
} TextDecoderLayer;

/* */
static void
free_text_layer(TextDecoderLayer *layer)
{
  if (layer == NULL) return;
  free_linear(layer->wq);
  free_linear(layer->wk);
  free_linear(layer->wv);
  free_linear(layer->wo);
  free_linear(layer->w1);
  free_linear(layer->w2);
  free_linear(layer->w3);
  free(layer->n1);
  free(layer->n2);
  free(layer->nq);
  free(layer->nk);
  free(layer->n3);
  free(layer->n4);
  free(layer);
}

/* One SigLIP (vision) encoder block */
typedef struct
{
  // ViT attention weights
  Linear *wq;  // (hidden_dim, hidden_dim).T
  Linear *wk;  // (hidden_dim, hidden_dim).T
  Linear *wv;  // (hidden_dim, hidden_dim).T
  Linear *wo;  // (hidden_dim, hidden_dim).T

  // ViT attention biases
  floatx *bq;  // (hidden_dim,)
  floatx *bk;  // (hidden_dim,)
  floatx *bv;  // (hidden_dim,)
  floatx *bo;  // (hidden_dim,)

  // ViT feedforward weights
  Linear *w1;  // (hidden_dim, mlp_dim).T
  Linear *w2;  // (hidden_dim, mlp_dim).T

  // ViT feedforward biases
  floatx *b1;  // (mlp_dim,)
  floatx *b2;  // (mlp_dim,)

  // ViT layernorm weights
  floatx *n1;  // (hidden_dim,)
  floatx *n2;  // (hidden_dim,)

  // ViT layernorm biases
  floatx *n1_b;  // (hidden_dim,)
  floatx *n2_b;  // (hidden_dim,)
} VisionEncoderLayer;

/* */
static void
free_vision_layer(VisionEncoderLayer *layer)
{
  if (layer == NULL) return;
  free_linear(layer->wq);
  free_linear(layer->wk);
  free_linear(layer->wv);
  free_linear(layer->wo);
  free(layer->bq);
  free(layer->bk);
  free(layer->bv);
  free(layer->bo);
  free_linear(layer->w1);
  free_linear(layer->w2);
  free(layer->b1);
  free(layer->b2);
  free(layer->n1);
  free(layer->n2);
  free(layer->n1_b);
  free(layer->n2_b);
  free(layer);
}

/* Vision encoder container */
typedef struct
{
  VisionConfig *config;

  floatx *patch_emb;      // (hidden_dim, 3, patch_size, patch_size)
  floatx *patch_emb_b;    // (hidden_dim,)
  Linear *pos_embedding;  // ((image_size / patch_size)^2, hidden_dim)

  VisionEncoderLayer **layers;

  floatx *post_norm;    // (hidden_dim,)
  floatx *post_norm_b;  // (hidden_dim,)
  floatx *norm;         // (hidden_dim,)
  Linear *proj;         // (hidden_dim, embed_dim).T
} VisionEncoder;

/* */
void
free_vision_encoder(VisionEncoder *enc)
{
  if (enc == NULL) return;

  free(enc->patch_emb);
  free(enc->patch_emb_b);
  free_linear(enc->pos_embedding);
  if (enc->layers != NULL)
  {
    for (int i = 0; i < enc->config->n_layers; i++)
    {
      free_vision_layer(enc->layers[i]);
    }
    free(enc->layers);
  }
  free(enc->config);
  free(enc->post_norm);
  free(enc->post_norm_b);
  free(enc->norm);
  free_linear(enc->proj);
  free(enc);
}

/* Text decoder container */
typedef struct
{
  TextConfig     *config;
  GemmaTokenizer *tokenizer;
  // (vocab_size, embed_dim), shared with lm_head (tied weights)
  Linear            *embedding;
  TextDecoderLayer **layers;
  floatx            *final_norm;  // (embed_dim,)
} TextDecoder;

/* */
void
free_text_decoder(TextDecoder *dec)
{
  if (dec == NULL) return;

  free_tokenizer(dec->tokenizer);
  free_linear(dec->embedding);
  if (dec->layers != NULL && dec->config != NULL)
  {
    for (int i = 0; i < dec->config->n_layers; i++)
    {
      free_text_layer(dec->layers[i]);
    }
    free(dec->layers);
  }
  free(dec->final_norm);
  free_text_config(dec->config);
  free(dec);
}

/* Top-level model container */
typedef struct
{
  TextDecoder   *decoder;
  VisionEncoder *encoder;
  bool           quant;  // W8A8
} GemmaModel;

/* */
void
free_gemma_model(GemmaModel *model)
{
  if (model == NULL) return;
  free_vision_encoder(model->encoder);
  free_text_decoder(model->decoder);
  free(model);
}

// Runtime buffers (allocated once, reused every step)

/**
 * SigLIP vision model runtime buffer
 * N       = n_patches
 * C       = hidden_dim
 * CM      = mlp_dim
 * tpi     = image_toks
 * C_embed = embed_dim
 */
typedef struct
{
  int8_t *x_i8;        // (N, C)
  floatx *x_scales;    // (N,)
  int8_t *mlp_i8;      // (N, CM)
  floatx *mlp_scales;  // (N)

  floatx *x;           // (N, C)
  floatx *resid;       // (N, C)
  floatx *xq;          // (N, C)
  floatx *xk;          // (N, C)
  floatx *xv;          // (N, C)
  floatx *att_out;     // (N, C)
  floatx *mlp_hidden;  // (N, CM)
  floatx *scores;      // (NH, N, N)
} VisionBuffer;

/* */
void
free_vision_buffer(VisionBuffer *buf, bool quant)
{
  if (buf == NULL) return;
  if (quant)
  {
    free(buf->x_i8);
    free(buf->x_scales);
    free(buf->mlp_i8);
    free(buf->mlp_scales);
  }
  free(buf->x);
  free(buf->resid);
  free(buf->xq);
  free(buf->xk);
  free(buf->xv);
  free(buf->att_out);
  free(buf->mlp_hidden);
  free(buf->scores);
  free(buf);
}

/* */
VisionBuffer *
malloc_vision_buffer(VisionConfig *vcfg, bool quant)
{
  VisionBuffer *buf = NULL;
  CALLOC(buf, 1, "vbuf", goto fail;);

  size_t C   = (size_t)vcfg->hidden_dim;
  size_t ppi = (size_t)vcfg->image_size / vcfg->patch_size;
  size_t N   = (size_t)ppi * ppi;
  size_t CM  = (size_t)vcfg->mlp_dim;
  size_t NH  = (size_t)vcfg->n_heads;

  if (quant)
  {
    // Quantization buffers
    MALLOC(buf->x_i8, N * C, "vbuf.x_i8", goto fail;);
    MALLOC(buf->mlp_i8, N * CM, "vbuf.mlp_i8", goto fail;);
    MALLOC(buf->x_scales, N, "vbuf.x_scales", goto fail;);
    MALLOC(buf->mlp_scales, N, "vbuf.mlp_scales", goto fail;);
  }

  MALLOC(buf->x, N * C, "vbuf.x", goto fail;);
  MALLOC(buf->resid, N * C, "vbuf.resid", goto fail;);
  MALLOC(buf->xq, N * C, "vbuf.q", goto fail;);
  MALLOC(buf->xk, N * C, "vbuf.k", goto fail;);
  MALLOC(buf->xv, N * C, "vbuf.v", goto fail;);
  MALLOC(buf->att_out, N * C, "vbuf.att_out", goto fail;);
  MALLOC(buf->mlp_hidden, N * CM, "vbuf.mlp_hidden", goto fail;);
  MALLOC(buf->scores, NH * N * N, "vbuf.scores", goto fail;);

  return buf;

fail:
  free_vision_buffer(buf, quant);
  return NULL;
}

/**
 * Gemma language model runtime buffer
 * T     = number of input tokens (used in prefilling)
 * L     = n_layers
 * C     = embed_dim
 * CM    = mlp_dim
 * NH    = n_heads
 * NH_kv = n_kv_heads
 */
typedef struct
{
  int cache_len;

  // Temporary quantized activations (when quant=true)
  int8_t *x_i8;       // ([T], C,)
  floatx *x_scales;   // ([T],)
  int8_t *xo_i8;      // (NH, [T], CH)
  floatx *xo_scales;  // ([T],)
  int8_t *xg_i8;      // ([T], CM,)
  floatx *xg_scales;  // ([T],)

  // Pre-computed cos/sin for RoPE
  floatx *csfreqs_slid;  // ([T], CH / 2, 2)
  floatx *csfreqs_full;  // ([T], CH / 2, 2)

  // Residual stream
  floatx *x;      // ([T], C,)
  floatx *resid;  // ([T], C,)

  // Attention buffers
  floatx *xq;        // (NH, [T], CH)
  floatx *xk;        // (NH_kv, [T], CH)
  floatx *xv;        // (NH_kv, CH, [T])
  floatx *xo;        // ([T], NH, CH)
  floatx *att;       // (NH, [T], cache_len)
  floatx *kv_cache;  // (L, 2, NH_kv, cache_len, CH)

  // MLP buffers
  floatx *xg;      // ([T], CM,)
  floatx *xu;      // ([T], CM,)
  floatx *logits;  // (vocab_size,)
} TextBuffer;

/* */
void
free_text_buffer(TextBuffer *buf, bool quant)
{
  if (buf == NULL) return;

  free(buf->x);
  free(buf->resid);
  free(buf->xq);
  free(buf->xk);
  free(buf->csfreqs_slid);
  free(buf->csfreqs_full);
  free(buf->xv);
  free(buf->xo);
  free(buf->att);
  free(buf->kv_cache);
  free(buf->xg);
  free(buf->xu);
  free(buf->logits);
  if (quant)
  {
    free(buf->x_i8);
    free(buf->x_scales);
    free(buf->xo_i8);
    free(buf->xo_scales);
    free(buf->xg_i8);
    free(buf->xg_scales);
  }
  free(buf);
}

/* */
TextBuffer *
malloc_text_buffer(TextConfig *cfg,
    VisionConfig              *vcfg,

    int  cache_len,
    int  chunk_size,
    bool enable_mm,
    bool quant)
{
  TextBuffer *buf = NULL;
  CALLOC(buf, 1, "buf", goto fail;);  // Init to all NULL

  buf->cache_len = cache_len;

  int C  = cfg->embed_dim;
  int L  = cfg->n_layers;
  int CH = cfg->head_dim;
  int NH = cfg->n_heads;
  int CM = cfg->mlp_dim;

  int Cq  = NH * CH;
  int Ckv = cfg->n_kv_heads * CH;

  MALLOC(buf->kv_cache, (size_t)L * 2 * (size_t)cache_len * (size_t)Ckv,
         "buf.kv_cache", goto fail;);
  MALLOC(buf->logits, cfg->vocab_size, "buf.logits", goto fail;);

  int mult = chunk_size;

  // Multimodal models need space for a whole image worth of tokens
  if (cfg->support_mm && enable_mm && vcfg != NULL)
  {
    mult = max(mult, cfg->image_toks);
  }

  if (quant)
  {
    // Quantization buffers
    MALLOC(buf->x_i8, mult * C, "buf.x_i8", goto fail;);
    MALLOC(buf->xo_i8, mult * Cq, "buf.xo_i8", goto fail;);
    MALLOC(buf->xg_i8, mult * CM, "buf.xg_i8", goto fail;);
    MALLOC(buf->x_scales, mult, "buf.x_scales", goto fail;);
    MALLOC(buf->xo_scales, mult, "buf.xo_scales", goto fail;);
    MALLOC(buf->xg_scales, mult, "buf.xg_scales", goto fail;);
  }

  MALLOC(buf->x, mult * C, "buf.x", goto fail;);
  MALLOC(buf->resid, mult * C, "buf.resid", goto fail;);
  MALLOC(buf->xq, mult * Cq, "buf.xq", goto fail;);
  MALLOC(buf->xk, mult * Ckv, "buf.xk", goto fail;);
  MALLOC(buf->csfreqs_slid, mult * CH, "buf.csfreqs_slid", goto fail;);
  MALLOC(buf->csfreqs_full, mult * CH, "buf.csfreqs_full", goto fail;);
  MALLOC(buf->xv, mult * Ckv, "buf.xv", goto fail;);
  MALLOC(buf->xo, mult * Cq, "buf.xo", goto fail;);
  MALLOC(buf->att, mult * NH * cache_len, "buf.att", goto fail;);
  MALLOC(buf->xg, mult * CM, "buf.xg", goto fail;);
  MALLOC(buf->xu, mult * CM, "buf.xu", goto fail;);

  return buf;

fail:
  free_text_buffer(buf, quant);
  return NULL;
}

/* Read the entire text & vision model from the custom binary file format */
GemmaModel *
read_model(const char *filename, bool enable_mm)
{
  // TODO: Rewrite this using mmap. Didn't know it exists back when I started
  // this project...
  FILE           *fp    = NULL;
  GemmaTokenizer *tok   = NULL;
  TextConfig     *cfg   = NULL;
  VisionEncoder  *enc   = NULL;
  VisionConfig   *vcfg  = NULL;
  TextDecoder    *dec   = NULL;
  GemmaModel     *model = NULL;

  char *att_layers_buf = NULL;

  fp = fopen(filename, "rb");
  if (fp == NULL)
  {
    fprintf(stderr, "error: failed to open file: %s\n", filename);
    goto fail;
  }

  CALLOC(tok, 1, "model.decoder.tokenizer", goto fail;);
  CALLOC(cfg, 1, "model.decoder.config", goto fail;);
  // Build the text model
  CALLOC(dec, 1, "model.decoder", goto fail;);
  dec->config    = cfg;
  dec->tokenizer = tok;
  // Build the full model
  CALLOC(model, 1, "model", goto fail;);
  model->decoder = dec;

  // Read the configs
  FGETC(cfg->n_layers, fp, "model.decoder.config.n_layers", goto fail;);
  FGETC(cfg->n_heads, fp, "model.decoder.config.n_heads", goto fail;);
  FGETC(cfg->n_kv_heads, fp, "model.decoder.config.n_kv_heads", goto fail;);

  READ_UINT16(cfg->head_dim, fp, "model.decoder.config.head_dim", goto fail;);
  READ_UINT16(cfg->embed_dim, fp, "model.decoder.config.embed_dim", goto fail;);
  READ_UINT16(cfg->mlp_dim, fp, "model.decoder.config.mlp_dim", goto fail;);
  READ_UINT16(cfg->q_scale, fp, "model.decoder.config.q_scale", goto fail;);
  READ_UINT16(cfg->slide_len, fp, "model.decoder.config.slide_len", goto fail;);
  READ_UINT16(
      cfg->image_toks, fp, "model.decoder.config.image_toks", goto fail;);
  READ_UINT32(
      cfg->max_seqlen, fp, "model.decoder.config.max_seqlen", goto fail;);
  READ_UINT32(
      cfg->vocab_size, fp, "model.decoder.config.vocab_size", goto fail;);

  READ_FP32(
      cfg->local_theta, fp, "model.decoder.config.local_theta", goto fail;);
  READ_FP32(
      cfg->global_theta, fp, "model.decoder.config.global_theta", goto fail;);
  READ_FP32(cfg->eps, fp, "model.decoder.config.eps", goto fail;);
  READ_FP32(
      cfg->att_softcap, fp, "model.decoder.config.att_softcap", goto fail;);
  READ_FP32(
      cfg->logit_softcap, fp, "model.decoder.config.logit_softcap", goto fail;);

  // Packed bit-field of which layers use sliding-window attention
  // A terrible terrible idea, wish I didn't do this
  int n_bytes;
  FGETC(n_bytes, fp, "model.decoder.config.att_layers", goto fail;);
  if (n_bytes * 8 < cfg->n_layers)
  {
    fprintf(stderr, "error: insufficient att_layers bytes\n");
    goto fail;
  }
  MALLOC(
      att_layers_buf, n_bytes, "model.decoder.config.att_layers", goto fail;);
  MALLOC(cfg->att_layers, cfg->n_layers, "model.decoder.config.att_layers",
         goto fail;);
  FREAD(att_layers_buf, n_bytes, fp, "model.decoder.config.att_layers",
        goto fail;);
  for (int i = 0; i < cfg->n_layers; i++)
  {
    int pos            = i;
    int byte_idx       = pos / 8;
    int bit_idx        = 7 - (pos % 8);
    cfg->att_layers[i] = (att_layers_buf[byte_idx] >> bit_idx) & 1;
  }
  free(att_layers_buf);
  att_layers_buf = NULL;

  // Extra feature flags packed into one byte
  int extra_flags;
  FGETC(extra_flags, fp, "model.decoder.config.extra_flags", goto fail;);
  cfg->support_mm   = (extra_flags & 16) == 16;
  cfg->qk_norm      = (extra_flags & 8) == 8;
  cfg->pre_mlp_norm = (extra_flags & 4) == 4;
  cfg->pst_mlp_norm = (extra_flags & 2) == 2;
  model->quant      = (extra_flags & 1);

  bool use_mm = cfg->support_mm && enable_mm;

  if (use_mm)
  {
    // Read vision config
    CALLOC(enc, 1, "model.encoder", goto fail;);
    CALLOC(vcfg, 1, "model.encoder.config", goto fail;);
    FGETC(vcfg->n_layers, fp, "model.encoder.config.n_layers", goto fail;);
    FGETC(vcfg->n_heads, fp, "model.encoder.config.n_heads", goto fail;);
    READ_UINT16(vcfg->mlp_dim, fp, "model.encoder.config.mlp_dim", goto fail;);
    READ_UINT16(
        vcfg->hidden_dim, fp, "model.encoder.config.hidden_dim", goto fail;);
    READ_UINT16(
        vcfg->image_size, fp, "model.encoder.config.image_size", goto fail;);
    READ_UINT16(
        vcfg->patch_size, fp, "model.encoder.config.patch_size", goto fail;);
    READ_FP32(vcfg->eps, fp, "model.encoder.config.eps", goto fail;);
    enc->config    = vcfg;
    model->encoder = enc;
  }
  else if (cfg->support_mm)
  {
    int   t0;
    float t1;
    // Skip vision config if user didn't ask for multimodal
    FGETC(t0, fp, "model.encoder.config.n_layers", goto fail;);
    FGETC(t0, fp, "model.encoder.config.n_heads", goto fail;);
    READ_UINT16(t0, fp, "model.encoder.config.mlp_dim", goto fail;);
    READ_UINT16(t0, fp, "model.encoder.config.hidden_dim", goto fail;);
    READ_UINT16(t0, fp, "model.encoder.config.image_size", goto fail;);
    READ_UINT16(t0, fp, "model.encoder.config.patch_size", goto fail;);
    READ_FP32(t1, fp, "model.encoder.config.eps", goto fail;);
  }

  // dtype
  int   offset = 0;
  char  dtype_buf[10];
  char *dtype;
  READ_STR(dtype, fp, dtype_buf, &offset, "dtype", goto fail;);
  if (strcmp(dtype, DTYPE_STR) != 0)
  {
    printf("dtype '%s' not supported\n", dtype);
    goto fail;
  }

  // Build vocabulary
  offset = 0;

  tok->vocab_size = cfg->vocab_size;
  if (cfg->support_mm)
  {
    tok->vocab_size++;
  }  // ++ for the <image_soft_token>
  int vocab_data_bytes = get_strarr_bytes(fp, tok->vocab_size);
  if (vocab_data_bytes == -1) goto fail;
  MALLOC(tok->vocab_data, vocab_data_bytes,
         "model.decoder.tokenizer.vocab_data", goto fail;);
  MALLOC(
      tok->vocab, tok->vocab_size, "model.decoder.tokenizer.vocab", goto fail;);
  MALLOC(tok->vocab_sorted, tok->vocab_size,
         "model.decoder.tokenizer.vocab_sorted", goto fail;);
  for (int i = 0; i < tok->vocab_size; i++)
  {
    char name[64];
    snprintf(name, sizeof(name), "model.decoder.tokenizer.vocab_data.%d", i);
    char *str;
    READ_STR(str, fp, tok->vocab_data, &offset, name, goto fail;);
    tok->vocab[i]            = str;
    tok->vocab_sorted[i].idx = i;
    tok->vocab_sorted[i].val = str;
  }
  qsort(tok->vocab_sorted, tok->vocab_size, sizeof(tok->vocab_sorted[0]),
      cmp_token);

  // Special tokens
  tok->bos = get_token_idx(tok, "<bos>");
  tok->eos = get_token_idx(tok, "<eos>");
  tok->sot = get_token_idx(tok, "<start_of_turn>");
  tok->eot = get_token_idx(tok, "<end_of_turn>");
  tok->soi = get_token_idx(tok, "<start_of_image>");
  tok->eoi = get_token_idx(tok, "<end_of_image>");
  tok->ist = get_token_idx(tok, "<image_soft_token>");

  // Build merges
  READ_UINT32(
      tok->n_merges, fp, "model.decoder.tokenizer.n_merges", goto fail;);
  MALLOC(
      tok->ranks, tok->n_merges, "model.decoder.tokenizer.ranks", goto fail;);
  int merge_bytes = get_strarr_bytes(fp, tok->n_merges * 2);
  if (merge_bytes == -1) goto fail;
  MALLOC(tok->merge_data, merge_bytes, "model.decoder.tokenizer.merge_data",
         goto fail;);

  offset = 0;
  for (int i = 0; i < tok->n_merges; i++)
  {
    char name0[64], name1[64];
    snprintf(
        name0, sizeof(name0), "model.decoder.tokenizer.merge_data.%d.0", i);
    snprintf(
        name1, sizeof(name1), "model.decoder.tokenizer.merge_data.%d.1", i);

    char *str1, *str2;
    READ_STR(str1, fp, tok->merge_data, &offset, name0, goto fail;);
    READ_STR(str2, fp, tok->merge_data, &offset, name1, goto fail;);

    tok->ranks[i].rank = i;
    tok->ranks[i].str1 = str1;
    tok->ranks[i].str2 = str2;
  }
  qsort(tok->ranks, tok->n_merges, sizeof(tok->ranks[0]), cmp_merge);

  /* The embedding shape is (vocab_size, embed_dim), but it uses per-tensor
   * quantization rather than per-channel like other weights. Gemma uses tied
   * weights, which means the final lm_head shares the same weights with the
   * embedding table, but transposed. So it becomes per-channel quantization in
   * the final lm_head.
   */
  int C = cfg->embed_dim;

  READ_LINEAR(dec->embedding, fp, C, cfg->vocab_size, model->quant,
              "model.decoder.embedding", goto fail;);
  CALLOC(dec->layers, cfg->n_layers, "model.decoder.layers",  // NOLINT
         goto fail;);

  int Cq  = cfg->n_heads * cfg->head_dim;
  int Ckv = cfg->n_kv_heads * cfg->head_dim;

  // Read all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    char layer_name[64];
    snprintf(layer_name, sizeof(layer_name), "model.decoder.layers.%d", l);
    TextDecoderLayer *layer = NULL;
    CALLOC(layer, 1, layer_name, goto fail;);

    char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
    snprintf(wq_name, sizeof(wq_name), "model.decoder.layers.%d.wq", l);
    snprintf(wk_name, sizeof(wk_name), "model.decoder.layers.%d.wk", l);
    snprintf(wv_name, sizeof(wv_name), "model.decoder.layers.%d.wv", l);
    snprintf(wo_name, sizeof(wo_name), "model.decoder.layers.%d.wo", l);

    // Attention weights
    READ_LINEAR(layer->wq, fp, C, Cq, model->quant, wq_name, goto fail;);
    READ_LINEAR(layer->wk, fp, C, Ckv, model->quant, wk_name, goto fail;);
    READ_LINEAR(layer->wv, fp, C, Ckv, model->quant, wv_name, goto fail;);
    READ_LINEAR(layer->wo, fp, Cq, C, model->quant, wo_name, goto fail;);

    if (cfg->qk_norm)
    {
      char nq_name[64], nk_name[64];
      snprintf(nq_name, sizeof(nq_name), "model.decoder.layers.%d.nq", l);
      snprintf(nk_name, sizeof(nk_name), "model.decoder.layers.%d.nk", l);
      READ_TENSOR(layer->nq, cfg->head_dim, fp, nq_name, goto fail;);
      READ_TENSOR(layer->nk, cfg->head_dim, fp, nk_name, goto fail;);
    }
    else
    {
      layer->nq = NULL;
      layer->nk = NULL;
    }

    char w1_name[64], w2_name[64], w3_name[64];
    snprintf(w1_name, sizeof(w1_name), "model.decoder.layers.%d.w1", l);
    snprintf(w2_name, sizeof(w2_name), "model.decoder.layers.%d.w2", l);
    snprintf(w3_name, sizeof(w3_name), "model.decoder.layers.%d.w3", l);

    // Feedforward weights
    READ_LINEAR(
        layer->w1, fp, C, cfg->mlp_dim, model->quant, w1_name, goto fail;);
    READ_LINEAR(
        layer->w2, fp, C, cfg->mlp_dim, model->quant, w2_name, goto fail;);
    READ_LINEAR(
        layer->w3, fp, cfg->mlp_dim, C, model->quant, w3_name, goto fail;);

    char n1_name[64], n2_name[64];
    snprintf(n1_name, sizeof(n1_name), "model.decoder.layers.%d.n1", l);
    snprintf(n2_name, sizeof(n2_name), "model.decoder.layers.%d.n2", l);

    // RMSNorm weights
    READ_TENSOR(layer->n1, C, fp, n1_name, goto fail;);
    READ_TENSOR(layer->n2, C, fp, n2_name, goto fail;);

    if (cfg->pre_mlp_norm)
    {
      char n3_name[64];
      snprintf(n3_name, sizeof(n3_name), "model.decoder.layers.%d.n3", l);
      READ_TENSOR(layer->n3, C, fp, n3_name, goto fail;);
    }
    else
    {
      layer->n3 = NULL;
    }
    if (cfg->pst_mlp_norm)
    {
      char n4_name[64];
      snprintf(n4_name, sizeof(n4_name), "model.decoder.layers.%d.n4", l);
      READ_TENSOR(layer->n4, C, fp, n4_name, goto fail;);
    }
    else
    {
      layer->n4 = NULL;
    }
    dec->layers[l] = layer;
  }
  READ_TENSOR(dec->final_norm, C, fp, "model.decoder.final_norm", goto fail;);

  if (use_mm)
  {
    int P  = vcfg->patch_size;
    int VC = vcfg->hidden_dim;

    READ_TENSOR(enc->patch_emb, VC * 3 * P * P, fp, "model.encoder.patch_emb",
                goto fail;);
    READ_TENSOR(
        enc->patch_emb_b, VC, fp, "model.encoder.patch_emb_b", goto fail;);
    int n_patches = vcfg->image_size / P;
    n_patches *= n_patches;
    // Same as here, the real shape is (n_patches, VC)
    READ_LINEAR(enc->pos_embedding, fp, VC, n_patches, model->quant,
                "model.encoder.pos_embedding", goto fail;);
    CALLOC(enc->layers, vcfg->n_layers, "model.encoder.layers",  // NOLINT
           goto fail;);

    // Read all the layers of ViT
    for (int l = 0; l < vcfg->n_layers; l++)
    {
      char layer_name[64];
      snprintf(layer_name, sizeof(layer_name), "model.encoder.layers.%d", l);
      VisionEncoderLayer *layer = NULL;
      CALLOC(layer, 1, layer_name, goto fail;);

      // First layernorm
      char n1_name[64], n1b_name[64];
      snprintf(n1_name, sizeof(n1_name), "model.encoder.layers.%d.n1", l);
      snprintf(n1b_name, sizeof(n1b_name), "model.encoder.layers.%d.n1_b", l);
      READ_TENSOR(layer->n1, VC, fp, n1_name, goto fail;);
      READ_TENSOR(layer->n1_b, VC, fp, n1b_name, goto fail;);

      // Attention weights
      char wq_name[64], wk_name[64], wv_name[64], wo_name[64];
      snprintf(wq_name, sizeof(wq_name), "model.encoder.layers.%d.wq", l);
      snprintf(wk_name, sizeof(wk_name), "model.encoder.layers.%d.wk", l);
      snprintf(wv_name, sizeof(wv_name), "model.encoder.layers.%d.wv", l);
      snprintf(wo_name, sizeof(wo_name), "model.encoder.layers.%d.wo", l);

      READ_LINEAR(layer->wq, fp, VC, VC, model->quant, wq_name, goto fail;);
      READ_LINEAR(layer->wk, fp, VC, VC, model->quant, wk_name, goto fail;);
      READ_LINEAR(layer->wv, fp, VC, VC, model->quant, wv_name, goto fail;);
      READ_LINEAR(layer->wo, fp, VC, VC, model->quant, wo_name, goto fail;);

      // Attention biases
      char bq_name[64], bk_name[64], bv_name[64], bo_name[64];
      snprintf(bq_name, sizeof(bq_name), "model.encoder.layers.%d.bq", l);
      snprintf(bk_name, sizeof(bk_name), "model.encoder.layers.%d.bk", l);
      snprintf(bv_name, sizeof(bv_name), "model.encoder.layers.%d.bv", l);
      snprintf(bo_name, sizeof(bo_name), "model.encoder.layers.%d.bo", l);

      READ_TENSOR(layer->bq, VC, fp, bq_name, goto fail;);
      READ_TENSOR(layer->bk, VC, fp, bk_name, goto fail;);
      READ_TENSOR(layer->bv, VC, fp, bv_name, goto fail;);
      READ_TENSOR(layer->bo, VC, fp, bo_name, goto fail;);

      // Second layernorm
      char n2_name[64], n2b_name[64];
      snprintf(n2_name, sizeof(n2_name), "model.encoder.layers.%d.n2", l);
      snprintf(n2b_name, sizeof(n2b_name), "model.encoder.layers.%d.n2_b", l);

      READ_TENSOR(layer->n2, VC, fp, n2_name, goto fail;);
      READ_TENSOR(layer->n2_b, VC, fp, n2b_name, goto fail;);

      // Feedforward weights
      char w1_name[64], w2_name[64];
      snprintf(w1_name, sizeof(w1_name), "model.encoder.layers.%d.w1", l);
      snprintf(w2_name, sizeof(w2_name), "model.encoder.layers.%d.w2", l);
      READ_LINEAR(
          layer->w1, fp, VC, vcfg->mlp_dim, model->quant, w1_name, goto fail;);
      READ_LINEAR(
          layer->w2, fp, vcfg->mlp_dim, VC, model->quant, w2_name, goto fail;);

      // Feedforward biases
      char b1_name[64], b2_name[64];
      snprintf(b1_name, sizeof(b1_name), "model.encoder.layers.%d.b1", l);
      snprintf(b2_name, sizeof(b2_name), "model.encoder.layers.%d.b2", l);
      READ_TENSOR(layer->b1, vcfg->mlp_dim, fp, b1_name, goto fail;);
      READ_TENSOR(layer->b2, VC, fp, b2_name, goto fail;);

      enc->layers[l] = layer;
    }

    // Post layernorm
    READ_TENSOR(enc->post_norm, VC, fp, "model.encoder.post_norm", goto fail;);
    READ_TENSOR(
        enc->post_norm_b, VC, fp, "model.encoder.post_norm_b", goto fail;);

    // Soft embedding RMSNorm
    READ_TENSOR(enc->norm, VC, fp, "model.encoder.norm", goto fail;);
    // Final projection
    READ_LINEAR(
        enc->proj, fp, VC, C, model->quant, "model.encoder.proj", goto fail;);
  }

  fclose(fp);
  return model;

fail:
  if (fp != NULL) fclose(fp);
  free(att_layers_buf);
  if (model != NULL)
  {
    free_gemma_model(model);
  }
  else
  {
    free_text_decoder(dec);
    free_vision_encoder(enc);
    free_text_config(cfg);
    free_tokenizer(tok);
  }
  return NULL;
}

// Math primitives

// Make sure the clamping is not optimized by compilers
#if defined(__GNUC__) && !defined(__clang__)  // GCC
#  define CLAMP_NO_FAST_MATH __attribute__((optimize("no-fast-math")))
#elif defined(__clang__)  // clang
#  define CLAMP_NO_FAST_MATH
#elif defined(_MSC_VER)  // MSVC
#  define CLAMP_NO_FAST_MATH
#else
#  define CLAMP_NO_FAST_MATH
#endif

#if defined(__clang__)
#  pragma float_control(precise, on, push)
#elif defined(_MSC_VER)
#  pragma float_control(push)
#  pragma float_control(precise, on)
#endif

/* */
static inline floatx CLAMP_NO_FAST_MATH
clamp_fpx(floatx v)
{
  return (floatx)fminf(FLOATX_MAX, fmaxf((float)-FLOATX_MAX, (float)v));
}

#if defined(__clang__) || defined(_MSC_VER)
#  pragma float_control(pop)
#endif

/* Gemma-style RMSNorm: (x * rsqrt(mean(x²) + eps)) * (weight + 1) */
static void
rmsnorm(floatx   *dst,
    const floatx *src,
    const floatx *weight,

    int   dim,
    float eps)
{
  float sqsum = 0.0f;
  #pragma omp simd reduction(+ : sqsum)
  for (int i = 0; i < dim; i++)
  {
    sqsum += (float)src[i] * (float)src[i];
  }
  float rms = 1.0f / sqrtf(sqsum / (float)dim + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)((float)src[i] * rms * (float)(weight[i] + 1));
  }
}

/* Classic LayerNorm used inside the SigLIP vision tower */
static void
layernorm(floatx *dst,
    const floatx *src,
    const floatx *weight,
    const floatx *bias,

    int   dim,
    float eps)
{
  float mean = 0.0f;
  #pragma omp simd reduction(+ : mean)
  for (int i = 0; i < dim; i++)
  {
    mean += (float)src[i];
  }
  mean /= (float)dim;

  float var = 0.0f;
  #pragma omp simd reduction(+ : var)
  for (int i = 0; i < dim; i++)
  {
    float diff = (float)src[i] - mean;
    var += diff * diff;
  }
  var /= (float)dim;

  float inv_std = 1.0f / sqrtf(var + eps);
  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)(((float)src[i] - mean) * inv_std * (float)weight[i] +
                      (float)bias[i]);
  }
}

/* Symmetric per-vector quantization into int8 [-127, 127]
 * Q(fpx src (dim,)) ~= int8 dst (dim,) * fpx vec_scale (1,) */
static floatx
quantize_act(int8_t *dst, const floatx *vec, int dim)
{
  floatx amax = 0.0f;

  #pragma omp simd reduction(max : amax)
  for (int d = 0; d < dim; d++)
  {
    floatx av = vec[d] >= 0 ? vec[d] : -vec[d];
    if (av > amax)
    {
      amax = av;
    }
  }

  float vec_scale = (float)amax > 0.0f ? (float)amax / 127.0f : 1.0f;

  for (int d = 0; d < dim; d++)
  {
    int q = (int)roundf((float)vec[d] / vec_scale);
    if (q > 127)
    {
      q = 127;
    }
    else if (q < -127)
    {
      q = -127;
    }
    dst[d] = (int8_t)q;
  }

  return (floatx)vec_scale;
}

/* Symmetric int8 quantization for a matrix of rows
 * Q(fpx src (m, n)) ~= int8 dst (m, n) * fpx dst_scales (m,) */
static void
quantize_acts(int8_t *restrict dst,
    floatx *restrict           dst_scales,
    const floatx *restrict     src,
    int                        src_stride,

    int m,
    int n,

    bool omp)
{
  if (src_stride == 0)
  {
    src_stride = n;
  }

  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst_scales[i] = quantize_act(dst + i * n, src + i * src_stride, n);
    }
    return;
  }
  for (int i = 0; i < m; i++)
  {
    dst_scales[i] = quantize_act(dst + i * n, src + i * src_stride, n);
  }
}

/* */
static inline floatx
gemv_fpx_row(const floatx *restrict vec,
    const floatx *restrict          mat,

    int n,
    int i)
{
  float sum = 0;
  #pragma omp simd reduction(+ : sum)
  for (int j = 0; j < n; j++)
  {
    sum += (float)mat[i * n + j] * (float)vec[j];
  }
  return (floatx)sum;
}

/* fpx matrix-vector multiply (NT)
 * fpx vec (n,) @ fpx mat (m, n).T = fpx dst (m,) */
static void
gemv_fpx(floatx *restrict  dst,
    const floatx *restrict mat,
    const floatx *restrict vec,

    int m,
    int n,

    bool omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst[i] = gemv_fpx_row(vec, mat, n, i);
    }
    return;
  }
  for (int i = 0; i < m; i++)
  {
    dst[i] = gemv_fpx_row(vec, mat, n, i);
  }
}

/* */
static inline floatx
gemv_int8_row(const int8_t *restrict vec,
    floatx                           vec_scale,
    const int8_t *restrict           mat,
    const floatx *restrict           mat_scales,

    int n,
    int i)
{
  int32_t sum = 0;
  #pragma omp simd reduction(+ : sum)
  for (int j = 0; j < n; j++)
  {
    sum += (int32_t)mat[i * n + j] * (int32_t)vec[j];
  }
  return (floatx)((float)sum * (float)vec_scale * (float)mat_scales[i]);
}

/* int8 matrix-vector multiply + dequant (NT)
 *   (int8 vec (n,)   * fpx vec_scale  (1,))
 * @ (int8 mat (m, n) * fpx mat_scales (m,)).T = fpx dst (m,) */
static void
gemv_int8(floatx *restrict dst,
    const int8_t *restrict mat,
    const floatx *restrict mat_scales,
    const int8_t *restrict vec,
    floatx                 vec_scale,

    int m,
    int n,

    bool omp)
{
  if (omp)
  {
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
      dst[i] = gemv_int8_row(vec, vec_scale, mat, mat_scales, n, i);
    }
    return;
  }

  for (int i = 0; i < m; i++)
  {
    dst[i] = gemv_int8_row(vec, vec_scale, mat, mat_scales, n, i);
  }
}

/* fpx matrix-vector multiply (NN)
 * fpx vec (n,) @ fpx mat (n, m) = fpx dst (m,) */
static void
gemv_fpx_nn(floatx *restrict dst,
    const floatx *restrict   vec,
    const floatx *restrict   mat,
    int                      mat_stride,

    int n,
    int m)
{
  if (mat_stride == 0) mat_stride = m;

  float acc[m];
  for (int j = 0; j < m; j++)
  {
    acc[j] = 0.0f;
  }

  for (int i = 0; i < n; i++)
  {
    float         v       = (float)vec[i];
    const floatx *mat_row = mat + i * mat_stride;
    #pragma omp simd
    for (int j = 0; j < m; j++)
    {
      acc[j] += v * (float)mat_row[j];
    }
  }

  for (int j = 0; j < m; j++)
  {
    dst[j] = (floatx)acc[j];
  }
}

/* Compute one MRxNR tile of dst = src @ mat.T, dot-product form.
 * `mat`/`src` rows are read with their natural strides; the tile itself
 * is dense (GEMM_NT_MR x GEMM_NT_NR), never partial. */
static inline void
gemm_fpx_kernel(floatx *restrict dst,
    int                          dst_stride,
    const floatx *restrict       mat,
    int                          mat_stride,
    const floatx *restrict       src,
    int                          src_stride,
    int                          k)
{
  float acc[GEMM_NT_MR][GEMM_NT_NR] = {{0}};

  const floatx *s[GEMM_NT_MR];
  const floatx *w[GEMM_NT_NR];

  for (int i = 0; i < GEMM_NT_MR; ++i)
  {
    s[i] = src + i * src_stride;
  }
  for (int j = 0; j < GEMM_NT_NR; ++j)
  {
    w[j] = mat + j * mat_stride;
  }

  for (int l = 0; l < k; ++l)
  {
    float a[GEMM_NT_MR];
    float b[GEMM_NT_NR];
    for (int i = 0; i < GEMM_NT_MR; ++i)
    {
      a[i] = (float)s[i][l];
    }
    for (int j = 0; j < GEMM_NT_NR; ++j)
    {
      b[j] = (float)w[j][l];
    }

    for (int i = 0; i < GEMM_NT_MR; ++i)
      for (int j = 0; j < GEMM_NT_NR; ++j)
      {
        acc[i][j] += a[i] * b[j];
      }
  }

  for (int i = 0; i < GEMM_NT_MR; ++i)
    for (int j = 0; j < GEMM_NT_NR; ++j)
    {
      dst[i * dst_stride + j] = (floatx)acc[i][j];
    }
}

/* Scalar fallback for the m%4 / n%4 remainder tiles (and for m or n < 4
 * outright, e.g. tiny prefill chunks). Same math as the original loop. */
static inline void
gemm_fpx_scalar(floatx *restrict dst,
    int                          dst_stride,
    const floatx *restrict       mat,
    int                          mat_stride,
    const floatx *restrict       src,
    int                          src_stride,

    int m,
    int n,
    int k)
{
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
    {
      float         sum     = 0.0f;
      const floatx *src_row = src + i * src_stride;
      const floatx *w_row   = mat + j * mat_stride;
      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
        sum += (float)src_row[l] * (float)w_row[l];
      dst[i * dst_stride + j] = (floatx)sum;
    }
}

/* fpx matrix-matrix multiply (NT)
 * fpx src (m, k) @ fpx mat.T (k, n) = fpx dst (m, n) */
static void
gemm_fpx(floatx *restrict  dst,
    int                    dst_stride,
    const floatx *restrict mat,
    int                    mat_stride,
    const floatx *restrict src,
    int                    src_stride,

    int  m,
    int  n,
    int  k,
    bool omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_NT_MR) * GEMM_NT_MR;
  int n_full = (n / GEMM_NT_NR) * GEMM_NT_NR;

  if (omp)
  {
    /* Loop order is jb (column panel of `mat`) outer, ib (row block of
     * `src`) inner: every tile computed while jb is fixed shares the same
     * 4 rows of `mat`, which is what actually needs to stay resident. */
    #pragma omp parallel for collapse(2)
    for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
      {
        gemm_fpx_kernel(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, src + ib * src_stride,
            src_stride, k);
      }
  }
  else
  {
    for (int jb = 0; jb < n_full; jb += GEMM_NT_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_NT_MR)
      {
        gemm_fpx_kernel(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, src + ib * src_stride,
            src_stride, k);
      }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_fpx_scalar(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        src + m_full * src_stride, src_stride, m - m_full, n, k);
  }
  if (n_full < n)
  {
    gemm_fpx_scalar(dst + n_full, dst_stride, mat + n_full * mat_stride,
        mat_stride, src, src_stride, m_full, n - n_full, k);
  }
}

/* Compute an `mr`x`n` (mr <= GEMM_NN_MR) block of dst = src @ mat.
 * `mat` is converted from floatx -> float once per `l` (into `row`) and
 * then reused across all `mr` accumulator rows, instead of being
 * re-read/re-converted once per row like the naive version. */
static inline void
gemm_fpx_nn_kernel(floatx *restrict dst,
    int                             dst_stride,
    const floatx *restrict          mat,
    int                             mat_stride,
    const floatx *restrict          src,
    int                             src_stride,

    int mr,
    int n,
    int k)
{
  float acc[GEMM_NN_MR][n];
  float row[n];
  for (int ii = 0; ii < mr; ++ii)
    memset(acc[ii], 0, n * sizeof(float));

  for (int l = 0; l < k; ++l)
  {
    const floatx *mat_row = mat + l * mat_stride;
    #pragma omp simd
    for (int j = 0; j < n; ++j)
    {
      row[j] = (float)mat_row[j];
    }

    for (int ii = 0; ii < mr; ++ii)
    {
      float a = (float)src[ii * src_stride + l];
      #pragma omp simd
      for (int j = 0; j < n; ++j)
      {
        acc[ii][j] += a * row[j];
      }
    }
  }

  for (int ii = 0; ii < mr; ++ii)
  {
    floatx *dst_row = dst + ii * dst_stride;
    for (int j = 0; j < n; ++j)
    {
      dst_row[j] = (floatx)acc[ii][j];
    }
  }
}

/* fpx matrix-matrix multiply (NN)
 * fpx src (m, k) @ fpx mat (k, n) = fpx dst (m, n) */
static void
gemm_fpx_nn(floatx *restrict dst,
    int                      dst_stride,
    const floatx *restrict   mat,
    int                      mat_stride,
    const floatx *restrict   src,
    int                      src_stride,

    int  m,
    int  n,
    int  k,
    bool omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = n;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_NN_MR) * GEMM_NN_MR;

  if (omp)
  {
    #pragma omp parallel for
    for (int ib = 0; ib < m_full; ib += GEMM_NN_MR)
    {
      gemm_fpx_nn_kernel(dst + ib * dst_stride, dst_stride, mat, mat_stride,
          src + ib * src_stride, src_stride, GEMM_NN_MR, n, k);
    }
  }
  else
  {
    for (int ib = 0; ib < m_full; ib += GEMM_NN_MR)
    {
      gemm_fpx_nn_kernel(dst + ib * dst_stride, dst_stride, mat, mat_stride,
          src + ib * src_stride, src_stride, GEMM_NN_MR, n, k);
    }
  }

  // Remainder: leftover rows (m % GEMM_NN_MR), still full width
  if (m_full < m)
  {
    gemm_fpx_nn_kernel(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        src + m_full * src_stride, src_stride, m - m_full, n, k);
  }
}

/* */
static inline void
gemm_int8_kernel(floatx *restrict dst,
    int                           dst_stride,
    const int8_t *restrict        mat,
    int                           mat_stride,
    const floatx *restrict        mat_scales,
    const int8_t *restrict        src,
    int                           src_stride,
    const floatx *restrict        src_scales,
    int                           k)
{
  int32_t acc[GEMM_I8_MR][GEMM_I8_NR] = {{0}};

  const int8_t *s[GEMM_I8_MR];
  const int8_t *w[GEMM_I8_NR];
  for (int i = 0; i < GEMM_I8_MR; ++i)
    s[i] = src + i * src_stride;
  for (int j = 0; j < GEMM_I8_NR; ++j)
    w[j] = mat + j * mat_stride;

  for (int l = 0; l < k; ++l)
  {
    int32_t a[GEMM_I8_MR];
    int32_t b[GEMM_I8_NR];
    for (int i = 0; i < GEMM_I8_MR; ++i)
      a[i] = s[i][l];
    for (int j = 0; j < GEMM_I8_NR; ++j)
      b[j] = w[j][l];

    for (int i = 0; i < GEMM_I8_MR; ++i)
    {
      for (int j = 0; j < GEMM_I8_NR; ++j)
      {
        acc[i][j] += a[i] * b[j];
      }
    }
  }

  for (int i = 0; i < GEMM_I8_MR; ++i)
  {
    float fscale = (float)src_scales[i];
    for (int j = 0; j < GEMM_I8_NR; ++j)
    {
      float val = (float)acc[i][j] * fscale * (float)mat_scales[j];
      dst[i * dst_stride + j] = (floatx)val;
    }
  }
}

/* Scalar fallback for the m%4 / n%4 remainder tiles. */
static inline void
gemm_int8_scalar(floatx *restrict dst,
    int                           dst_stride,
    const int8_t *restrict        mat,
    int                           mat_stride,
    const floatx *restrict        mat_scales,
    const int8_t *restrict        src,
    int                           src_stride,
    const floatx *restrict        src_scales,

    int m,
    int n,
    int k)
{
  for (int i = 0; i < m; i++)
  {
    float fscale = (float)src_scales[i];
    for (int j = 0; j < n; j++)
    {
      int32_t       sum     = 0;
      const int8_t *src_row = src + i * src_stride;
      const int8_t *mat_row = mat + j * mat_stride;
      #pragma omp simd reduction(+ : sum)
      for (int l = 0; l < k; l++)
      {
        sum += (int32_t)src_row[l] * (int32_t)mat_row[l];
      }
      float val               = (float)sum * fscale * (float)mat_scales[j];
      dst[i * dst_stride + j] = (floatx)val;
    }
  }
}

/* int8 matrix-matrix multiply (NT) + dequant
 *
 *   (int8 src (m, k) * fpx src_scales (m,))
 * @ (int8 mat (n, k) * fpx mat_scales (n,)).T = (fpx dst (m, n)) */
static void
gemm_int8(floatx *restrict dst,
    int                    dst_stride,
    const int8_t *restrict mat,
    int                    mat_stride,
    const floatx *restrict mat_scales,
    const int8_t *restrict src,
    int                    src_stride,
    const floatx *restrict src_scales,

    int  m,
    int  n,
    int  k,
    bool omp)
{
  if (dst_stride == 0) dst_stride = n;
  if (mat_stride == 0) mat_stride = k;
  if (src_stride == 0) src_stride = k;

  int m_full = (m / GEMM_I8_MR) * GEMM_I8_MR;
  int n_full = (n / GEMM_I8_NR) * GEMM_I8_NR;

  if (omp)
  {
    #pragma omp parallel for collapse(2)
    for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
      {
        gemm_int8_kernel(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, mat_scales + jb,
            src + ib * src_stride, src_stride, src_scales + ib, k);
      }
  }
  else
  {
    for (int jb = 0; jb < n_full; jb += GEMM_I8_NR)
      for (int ib = 0; ib < m_full; ib += GEMM_I8_MR)
      {
        gemm_int8_kernel(dst + ib * dst_stride + jb, dst_stride,
            mat + jb * mat_stride, mat_stride, mat_scales + jb,
            src + ib * src_stride, src_stride, src_scales + ib, k);
      }
  }

  // Remainder: leftover rows (full width) + leftover columns (remaining
  // height only, so the (m_full:m, n_full:n) corner isn't done twice)
  if (m_full < m)
  {
    gemm_int8_scalar(dst + m_full * dst_stride, dst_stride, mat, mat_stride,
        mat_scales, src + m_full * src_stride, src_stride, src_scales + m_full,
        m - m_full, n, k);
  }
  if (n_full < n)
  {
    gemm_int8_scalar(dst + n_full, dst_stride, mat + n_full * mat_stride,
        mat_stride, mat_scales + n_full, src, src_stride, src_scales, m_full,
        n - n_full, k);
  }
}

/* */
static void
softmax(floatx *dst, const floatx *src, int dim)
{
  floatx max = -(floatx)INFINITY;
  for (int i = 0; i < dim; i++)
  {
    if (src[i] > max)
    {
      max = src[i];
    }
  }

  float expsum = 0.0f;
  for (int i = 0; i < dim; i++)
  {
    float val = expf((float)(src[i] - max));
    dst[i]    = (floatx)val;
    expsum += val;
  }

  #pragma omp simd
  for (int i = 0; i < dim; i++)
  {
    dst[i] = (floatx)((float)dst[i] / expsum);
  }
}

/* */
floatx *
prepare_image(const char *path, int image_size)
{
  // Load the image
  int            rows, cols, channels;
  unsigned char *img = stbi_load(path, &cols, &rows, &channels, 3);
  if (img == NULL)
  {
    return NULL;
  }

  // Resize the image
  unsigned char *rsz = (unsigned char *)stbir_resize(img, cols, rows, 0, NULL,
      image_size, image_size, 0, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP,
      STBIR_FILTER_CATMULLROM);
  stbi_image_free(img);
  if (rsz == NULL)
  {
    return NULL;
  }

  floatx *out;
  MALLOC(out, image_size * image_size * 3, path, {
    free(rsz);
    return NULL;
  });
  for (int i = 0; i < image_size * image_size * 3; i++)
  {
    out[i] = (rsz[i] / 255.0f - 0.5f) / 0.5f;
  }

  free(rsz);
  return out;
}

// Forward passes

/* Vision forward pass (SigLIP)
 * img: (img_sz, img_sz, 3) */
int
forward_vision(VisionEncoder *enc,

    TextConfig   *cfg,
    TextBuffer   *buf,
    VisionBuffer *vbuf,
    const floatx *img,
    bool          quant)
{
  VisionConfig *vcfg = enc->config;

  int C        = vcfg->hidden_dim;
  int P        = vcfg->patch_size;
  int img_sz   = vcfg->image_size;
  int ppi      = img_sz / P;
  int N        = ppi * ppi;
  int tpi      = cfg->image_toks;
  int side_len = (int)roundf(sqrtf((float)tpi));
  int K        = (img_sz / vcfg->patch_size) / side_len;

  int CH     = C / vcfg->n_heads;
  int CM     = vcfg->mlp_dim;
  int in_dim = 3 * P * P;

  // Patch Embedding

  // Iterate over all the patch
  // Can be further optimized by reusing GEMM, but this part is executed only
  // once per call, the cost is acceptable
  #pragma omp parallel for collapse(2)
  for (int oy = 0; oy < ppi; oy++)
    for (int ox = 0; ox < ppi; ox++)
    {
      int patch_idx = oy * ppi + ox;

      // Compute the embed vector for the current patch
      for (int oc = 0; oc < C; oc++)
      {
        float sum = 0.0f;

        // equivalent to Conv2d(
        //   in_channels=3, out_channels=C, kernal_size=P, stride=P, bias=True)
        #pragma omp simd collapse(3)
        for (int py = 0; py < P; py++)
          for (int px = 0; px < P; px++)
            for (int c = 0; c < 3; c++)
            {
              // img[c, oy*P + py, ox*P + px]
              int in_idx = ((oy * P + py) * img_sz + (ox * P + px)) * 3 + c;
              // patch_emb[oc, c, py, px]
              int w_idx = oc * in_dim + c * P * P + py * P + px;
              sum += (float)enc->patch_emb[w_idx] * (float)img[in_idx];
            }

        vbuf->x[patch_idx * C + oc] =
            clamp_fpx((floatx)(sum + (float)enc->patch_emb_b[oc]));
      }
    }

  if (g_interrupted) return 1;

  // Position Embedding
  if (!quant)
  {
    #pragma omp simd
    for (int d = 0; d < N * C; d++)
    {
      vbuf->x[d] = clamp_fpx(vbuf->x[d] + enc->pos_embedding->fpx[d]);
    }
  }
  else
  {
    // Dequantize per row
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        float scale        = (float)enc->pos_embedding->i8.scales[i];
        float val          = (float)enc->pos_embedding->i8.q[i * C + j] * scale;
        vbuf->x[i * C + j] = clamp_fpx(vbuf->x[i * C + j] + (floatx)val);
      }
  }

  if (g_interrupted) return 1;

  // Encoder Layers
  for (int l = 0; l < vcfg->n_layers; l++)
  {
    VisionEncoderLayer *layer = enc->layers[l];

    memcpy(vbuf->resid, vbuf->x, N * C * sizeof(floatx));
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      floatx *row = vbuf->x + i * C;
      layernorm(row, row, layer->n1, layer->n1_b, C, vcfg->eps);
    }
    if (g_interrupted) return 1;

    // QKV projections
    if (!quant)
    {
      gemm_fpx(vbuf->xq, 0, layer->wq->fpx, 0, vbuf->x, 0, N, C, C, true);
      gemm_fpx(vbuf->xk, 0, layer->wk->fpx, 0, vbuf->x, 0, N, C, C, true);
      gemm_fpx(vbuf->xv, 0, layer->wv->fpx, 0, vbuf->x, 0, N, C, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, N, C, true);
      gemm_int8(vbuf->xq, 0, layer->wq->i8.q, 0, layer->wq->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
      gemm_int8(vbuf->xk, 0, layer->wk->i8.q, 0, layer->wk->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
      gemm_int8(vbuf->xv, 0, layer->wv->i8.q, 0, layer->wv->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
    }
    if (g_interrupted) return 1;

    // Add biases
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        int idx = i * C + j;
        vbuf->xq[idx] += layer->bq[j];
        vbuf->xk[idx] += layer->bk[j];
        vbuf->xv[idx] += layer->bv[j];
      }
    if (g_interrupted) return 1;

    // Attention
    memset(vbuf->att_out, 0, N * C * sizeof(floatx));
    float scale = 1.0f / sqrtf((float)CH);

    #pragma omp parallel for
    for (int h = 0; h < vcfg->n_heads; h++)
    {
      floatx *scores = vbuf->scores + h * N * N;  // (N, N) for this head

      // scores = Q @ K^T * scale
      gemm_fpx(scores, /*dst_stride=*/N, vbuf->xk + h * CH, /*mat_stride=*/C,
          vbuf->xq + h * CH, /*src_stride=*/C, N, N, CH, false);

      // Apply scale and softmax per row
      for (int i = 0; i < N; i++)
      {
        floatx *row = scores + i * N;
        #pragma omp simd
        for (int j = 0; j < N; j++)
        {
          row[j] *= scale;
        }
        softmax(row, row, N);
      }

      // Weighted sum of values: out = scores @ V
      gemm_fpx_nn(vbuf->att_out + h * CH, /*dst_stride=*/C, vbuf->xv + h * CH,
          /*mat_stride=*/C, scores, /*src_stride=*/N, N, CH, N, false);
    }
    if (g_interrupted) return 1;

    // Output projection
    if (!quant)
    {
      gemm_fpx(vbuf->x, 0, layer->wo->fpx, 0, vbuf->att_out, 0, N, C, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->att_out, 0, N, C, true);
      gemm_int8(vbuf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, C, C, true);
    }
    if (g_interrupted) return 1;

    // Add output bias
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->bo[j];
      }
    if (g_interrupted) return 1;

    // Residual connection
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (g_interrupted) return 1;

    memcpy(vbuf->resid, vbuf->x, N * C * sizeof(floatx));
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      floatx *row = vbuf->x + i * C;
      layernorm(row, row, layer->n2, layer->n2_b, C, vcfg->eps);
    }
    if (g_interrupted) return 1;

    // x @ fc1 = mlp_hidden
    if (!quant)
    {
      gemm_fpx(
          vbuf->mlp_hidden, 0, layer->w1->fpx, 0, vbuf->x, 0, N, CM, C, true);
    }
    else
    {
      quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, N, C, true);
      gemm_int8(vbuf->mlp_hidden, 0, layer->w1->i8.q, 0, layer->w1->i8.scales,
          vbuf->x_i8, 0, vbuf->x_scales, N, CM, C, true);
    }
    if (g_interrupted) return 1;

    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < CM; j++)
      {
        // Apply fc1 biases
        float val = (float)vbuf->mlp_hidden[i * CM + j] + (float)layer->b1[j];
        // GELU tanh approximation
        float c = 0.79788456080287f;
        val     = 0.5f * val *
              (1.0f + tanhf(c * (val + 0.044715f * val * val * val)));
        vbuf->mlp_hidden[i * CM + j] = (floatx)val;
      }
    if (g_interrupted) return 1;

    // mlp_hidden @ fc2 = x
    if (!quant)
    {
      gemm_fpx(
          vbuf->x, 0, layer->w2->fpx, 0, vbuf->mlp_hidden, 0, N, C, CM, true);
    }
    else
    {
      quantize_acts(
          vbuf->mlp_i8, vbuf->mlp_scales, vbuf->mlp_hidden, 0, N, CM, true);
      gemm_int8(vbuf->x, 0, layer->w2->i8.q, 0, layer->w2->i8.scales,
          vbuf->mlp_i8, 0, vbuf->mlp_scales, N, C, CM, true);
    }
    if (g_interrupted) return 1;

    // x += b2
    #pragma omp simd collapse(2)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < C; j++)
      {
        vbuf->x[i * C + j] += layer->b2[j];
      }

    if (g_interrupted) return 1;

    // Residual connection
    #pragma omp parallel for
    for (int i = 0; i < N * C; i++)
    {
      vbuf->x[i] = clamp_fpx(vbuf->x[i] + vbuf->resid[i]);
    }
    if (g_interrupted) return 1;
  }

  // Post-norm + average pooling down to image_toks tokens
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    floatx *row = vbuf->x + i * C;
    layernorm(row, row, enc->post_norm, enc->post_norm_b, C, vcfg->eps);
  }
  if (g_interrupted) return 1;

  // Average pooling (AvgPool2d(kernel_size=K, stride=K))

  /* Be careful that this loop must remain serial, in-place pooling writes
   * results to the front of vbuf->x, which overlaps with input data needed by
   * other output positions. Parallelizing this loop introduces a read/write
   * race condition. */
  for (int oy = 0; oy < side_len; oy++)
    for (int ox = 0; ox < side_len; ox++)
    {
      int out_idx = (oy * side_len + ox) * C;
      for (int d = 0; d < C; d++)
      {
        float sum = 0.0f;
        for (int ky = 0; ky < K; ky++)
        {
          #pragma omp simd reduction(+ : sum)
          for (int kx = 0; kx < K; kx++)
          {
            int py        = oy * K + ky;
            int px        = ox * K + kx;
            int token_idx = (py * ppi + px) * C + d;
            sum += (float)vbuf->x[token_idx];
          }
        }
        vbuf->x[out_idx + d] = (floatx)(sum / (K * K));
      }
    }

  if (g_interrupted) return 1;
  // buf->x now becomes (tpi, C)

  // Final RMSNorm
  #pragma omp parallel for
  for (int i = 0; i < tpi; i++)
  {
    floatx *row = vbuf->x + i * C;
    rmsnorm(row, row, enc->norm, C, vcfg->eps);
  }
  if (g_interrupted) return 1;

  // Final projection into language model embedding space
  // x (tpi, embed_dim) = x (tpi, C) @ proj (C, embed_dim)
  if (!quant)
  {
    gemm_fpx(
        buf->x, 0, enc->proj->fpx, 0, vbuf->x, 0, tpi, cfg->embed_dim, C, true);
  }
  else
  {
    quantize_acts(vbuf->x_i8, vbuf->x_scales, vbuf->x, 0, tpi, C, true);
    gemm_int8(buf->x, 0, enc->proj->i8.q, 0, enc->proj->i8.scales, vbuf->x_i8,
        0, vbuf->x_scales, tpi, cfg->embed_dim, C, true);
  }
  if (g_interrupted) return 1;

  return 0;
}

/* Language model forward (one token) */
int
forward_text_decode(
    TextDecoder *dec, TextBuffer *buf, int pos, bool quant, bool compute_logits)
{
  TextConfig *cfg = dec->config;

  int C     = cfg->embed_dim;
  int NH    = cfg->n_heads;
  int NH_kv = cfg->n_kv_heads;
  int CH    = cfg->head_dim;
  int Cq    = NH * CH;
  int Ckv   = NH_kv * CH;

  int CH_half = CH / 2;

  if (pos >= buf->cache_len)
  {
    fprintf(stderr, "\nerror: KV Cache is full\n");
    return 1;
  }

  // Precompute cos & sin for all frequencies (used in RoPE)
  for (int d = 0; d < CH_half; d++)
  {
    float freq;
    float e = (float)(-2 * d) / (float)CH;  // exponent

    // Rotation angles for sliding window attentions
    freq                         = powf(cfg->local_theta, e);
    buf->csfreqs_slid[d * 2]     = (floatx)cosf(freq * (float)pos);
    buf->csfreqs_slid[d * 2 + 1] = (floatx)sinf(freq * (float)pos);

    // Rotation angles for full attentions
    freq                         = powf(cfg->global_theta, e);
    buf->csfreqs_full[d * 2]     = (floatx)cosf(freq * (float)pos);
    buf->csfreqs_full[d * 2 + 1] = (floatx)sinf(freq * (float)pos);
  }
  if (g_interrupted) return 1;

  // Forward all the layers
  for (int l = 0; l < cfg->n_layers; l++)
  {
    TextDecoderLayer *layer = dec->layers[l];

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    rmsnorm(buf->x, buf->x, layer->n1, C, cfg->eps);
    if (g_interrupted) return 1;

    // The attention block
    if (!quant)
    {
      gemv_fpx(buf->xq, layer->wq->fpx, buf->x, Cq, C, true);  // (NH, CH)
      // (NH_kv, CH)
      gemv_fpx(buf->xk, layer->wk->fpx, buf->x, Ckv, C, true);
      gemv_fpx(buf->xv, layer->wv->fpx, buf->x, Ckv, C, true);
    }
    else
    {
      floatx x_scale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->xq, layer->wq->i8.q, layer->wq->i8.scales, buf->x_i8,
          x_scale, Cq, C, true);
      gemv_int8(buf->xk, layer->wk->i8.q, layer->wk->i8.scales, buf->x_i8,
          x_scale, Ckv, C, true);
      gemv_int8(buf->xv, layer->wv->i8.q, layer->wv->i8.scales, buf->x_i8,
          x_scale, Ckv, C, true);
    }
    if (g_interrupted) return 1;

    if (cfg->qk_norm)
    {
      // Query RMSNorm
      for (int h = 0; h < NH; h++)
      {
        floatx *xq_head = buf->xq + h * CH;
        // Use the non-threading version here since we are running this over
        // every head
        rmsnorm(xq_head, xq_head, layer->nq, CH, cfg->eps);
      }
      // Key RMSNorm
      for (int h = 0; h < NH_kv; h++)
      {
        floatx *xk_head = buf->xk + h * CH;
        rmsnorm(xk_head, xk_head, layer->nk, CH, cfg->eps);
      }
    }
    if (g_interrupted) return 1;

    bool    is_local = cfg->att_layers[l];
    floatx *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;

    // Apply RoPE to queries & keys
    for (int idx = 0; idx < NH + NH_kv; idx++)
    {
      floatx *data;
      if (idx < NH)
      {
        data = buf->xq + idx * CH;  // Apply to queries
      }
      else
      {
        data = buf->xk + (idx - NH) * CH;  // Apply to keys
      }

      for (int d = 0; d < CH_half; d++)
      {
        float cfr = (float)freqs_cs[2 * d];
        float sfr = (float)freqs_cs[2 * d + 1];
        float a   = (float)data[d];            // Index in the first half vector
        float b   = (float)data[d + CH_half];  // ... second half vector

        data[d]           = (floatx)(a * cfr - b * sfr);
        data[d + CH_half] = (floatx)(a * sfr + b * cfr);
      }
    }
    if (g_interrupted) return 1;

    // (NH_kv, cache_len, CH)
    floatx *k_cache = buf->kv_cache + l * 2 * buf->cache_len * Ckv;
    floatx *v_cache = k_cache + buf->cache_len * Ckv;

    // Write to kv_cache
    for (int h = 0; h < NH_kv; h++)
    {
      floatx *xk_head = k_cache + h * buf->cache_len * CH + pos * CH;
      floatx *xv_head = v_cache + h * buf->cache_len * CH + pos * CH;
      memcpy(xk_head, buf->xk + h * CH, CH * sizeof(*buf->xk));
      memcpy(xv_head, buf->xv + h * CH, CH * sizeof(*buf->xv));
    }
    if (g_interrupted) return 1;

    // Sliding-window (true) or full attention (false)?
    bool local_att = is_local && pos >= cfg->slide_len;
    // Starting position of kv_cache
    int spos   = local_att ? (pos + 1 - cfg->slide_len) : 0;
    int attlen = pos + 1 - spos;  // Include the current pos

    floatx att_scale = (floatx)(1.0f / sqrtf((float)cfg->q_scale));

    // Iterate over all the attention heads
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int h_kv = h * NH_kv / NH;  // GQA mapping

      floatx *xq_head = buf->xq + h * CH;  // xq[h, :]
      // k_cache[h_kv, spos:, :]
      floatx *xk_head = k_cache + h_kv * buf->cache_len * CH + spos * CH;
      // att[h, spos:]
      floatx *att_head = buf->att + h * attlen + spos;

      // Compute dot product of the current query across all the keys
      gemv_fpx(att_head, xk_head, xq_head, attlen, CH, false);
      for (int t = 0; t < attlen; t++)
      {
        att_head[t] *= att_scale;
      }

      // Attention score softcapping
      if (cfg->att_softcap != 0.0f)
      {
        for (int t = 0; t < attlen; t++)
        {
          float val   = (float)att_head[t] / cfg->att_softcap;
          att_head[t] = (floatx)(tanhf(val) * cfg->att_softcap);
        }
      }

      // Softmax
      softmax(att_head, att_head, attlen);

      // Compute output as weighted sum of values
      // v_cache[h_kv, spos:, :]
      floatx *xv_head = v_cache + h_kv * buf->cache_len * CH + spos * CH;
      floatx *xo_head = buf->xo + h * CH;

      // xo_head (CH,) = att_head (attlen,) @ xv_head (attlen, CH)
      gemv_fpx_nn(xo_head, att_head, xv_head, 0, attlen, CH);
    }
    if (g_interrupted) return 1;

    // Output projection maps xo back to x
    // x (C,) = xo (CH,) @ wo.T (CH, C)
    if (!quant)
    {
      gemv_fpx(buf->x, layer->wo->fpx, buf->xo, C, Cq, true);
    }
    else
    {
      floatx xo_scale = quantize_act(buf->xo_i8, buf->xo, Cq);
      gemv_int8(buf->x, layer->wo->i8.q, layer->wo->i8.scales, buf->xo_i8,
          xo_scale, C, Cq, true);
    }
    if (g_interrupted) return 1;

    rmsnorm(buf->x, buf->x, layer->n2, C, cfg->eps);
    if (g_interrupted) return 1;

    // Combine the residual stream
    floatx *restrict x     = buf->x;
    floatx *restrict resid = buf->resid;
    for (int d = 0; d < C; d++)
    {
      /* Sometimes the residual stream accumulates huge values on certain
       * channels, especially in pretrained/bigger models (Sun et al., 2024,
       * https://arxiv.org/abs/2402.17762). It works fine in fp32 or bf16, but
       * it can easily overflow fp16 and become inf, causing all the activations
       * turning into nan after the next RMSNorm, so we need to clamp it. */

      /* NOTE: Actually this should never trigger now since I added activation
       * scalers afterwards (see export.py), the clamp here is more of a
       * last-resort safety net. */

      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;

    memcpy(buf->resid, buf->x, C * sizeof(*buf->x));

    // Pre feedforward RMSNorm
    if (cfg->pre_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n3, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    // MLP feedforward layer (SwiGLU-style)
    if (!quant)
    {
      gemv_fpx(buf->xu, layer->w1->fpx, buf->x, cfg->mlp_dim, C, true);
      gemv_fpx(buf->xg, layer->w2->fpx, buf->x, cfg->mlp_dim, C, true);
    }
    else
    {
      floatx x_scale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->xg, layer->w2->i8.q, layer->w2->i8.scales, buf->x_i8,
          x_scale, cfg->mlp_dim, C, true);
      gemv_int8(buf->xu, layer->w1->i8.q, layer->w1->i8.scales, buf->x_i8,
          x_scale, cfg->mlp_dim, C, true);
    }
    if (g_interrupted) return 1;

    // GELU gate
    #pragma omp parallel for
    for (int d = 0; d < cfg->mlp_dim; d++)
    {
      // Tanh approximation of GELU
      float x    = (float)buf->xg[d];
      float c    = 0.79788456080287f;  // = sqrt(2 / pi)
      x          = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)));
      buf->xg[d] = (floatx)x;
      buf->xg[d] *= buf->xu[d];  // Fuse xg * xu into xg
    }
    if (g_interrupted) return 1;

    if (!quant)
    {
      gemv_fpx(buf->x, layer->w3->fpx, buf->xg, C, cfg->mlp_dim, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->xg_i8, buf->xg, cfg->mlp_dim);
      gemv_int8(buf->x, layer->w3->i8.q, layer->w3->i8.scales, buf->xg_i8,
          xscale, C, cfg->mlp_dim, true);
    }
    if (g_interrupted) return 1;

    // Post feedforward RMSNorm
    if (cfg->pst_mlp_norm)
    {
      rmsnorm(buf->x, buf->x, layer->n4, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    // Second residual
    x     = buf->x;
    resid = buf->resid;
    for (int d = 0; d < C; d++)
    {
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;
  }

  // Final RMSNorm
  rmsnorm(buf->x, buf->x, dec->final_norm, C, cfg->eps);
  if (g_interrupted) return 1;

  // Compute logits (tied embedding)
  if (compute_logits)
  {
    if (!quant)
    {
      gemv_fpx(
          buf->logits, dec->embedding->fpx, buf->x, cfg->vocab_size, C, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->x_i8, buf->x, C);
      gemv_int8(buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales,
          buf->x_i8, xscale, cfg->vocab_size, C, true);
    }
    if (g_interrupted) return 1;

    // Optional logit softcapping
    if (cfg->logit_softcap != 0.0f)
    {
      for (int d = 0; d < cfg->vocab_size; d++)
      {
        float val      = (float)buf->logits[d] / cfg->logit_softcap;
        buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
      }
    }
    if (g_interrupted) return 1;
  }
  return 0;
}

/* Language model forward (a chunk of tokens) */
static int
forward_text_chunk(TextDecoder *dec,
    TextBuffer                 *buf,

    int  spos,
    int  T,
    bool mask,
    bool quant,
    bool compute_logits)
{
  TextConfig *cfg = dec->config;

  if (quant && (buf->x_scales == NULL || buf->xo_scales == NULL ||
                   buf->xg_scales == NULL))
  {
    fprintf(stderr, "\nerror: scale buffers are not allocated\n");
    return 1;
  }

  int C       = cfg->embed_dim;
  int NH      = cfg->n_heads;
  int NH_kv   = cfg->n_kv_heads;
  int CH      = cfg->head_dim;
  int Cq      = NH * CH;
  int Ckv     = NH_kv * CH;
  int CH_half = CH / 2;

  int epos = spos + T - 1;

  if (epos >= buf->cache_len)
  {
    fprintf(stderr, "\nerror: KV Cache is full\n");
    return 1;
  }

  // Precompute RoPE angles
  #pragma omp parallel for
  for (int t = 0; t < T; t++)
  {
    int pos = spos + t;
    int off = t * CH;

    for (int d = 0; d < CH_half; d++)
    {
      float freq;
      float e = (float)(-2 * d) / (float)CH;

      // Sliding window angles
      freq                               = powf(cfg->local_theta, e);
      buf->csfreqs_slid[off + d * 2]     = (floatx)cosf(freq * (float)pos);
      buf->csfreqs_slid[off + d * 2 + 1] = (floatx)sinf(freq * (float)pos);

      // Full attention angles
      freq                               = powf(cfg->global_theta, e);
      buf->csfreqs_full[off + d * 2]     = (floatx)cosf(freq * (float)pos);
      buf->csfreqs_full[off + d * 2 + 1] = (floatx)sinf(freq * (float)pos);
    }
  }
  if (g_interrupted) return 1;

  floatx att_scale = (floatx)(1.0f / sqrtf((float)cfg->q_scale));

  for (int l = 0; l < cfg->n_layers; l++)
  {
    TextDecoderLayer *layer = dec->layers[l];

    memcpy(buf->resid, buf->x, T * C * sizeof(*buf->x));

    #pragma omp parallel for
    for (int t = 0; t < T; t++)
    {
      floatx *x_row = buf->x + t * C;
      rmsnorm(x_row, x_row, layer->n1, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    // The attention block

    if (quant)
    {
      quantize_acts(buf->x_i8, buf->x_scales, buf->x, 0, T, C, true);
    }

    // Compute xq & xk & xv
    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      floatx *xq_head = buf->xq + h * T * CH;  // (T, CH)
      if (!quant)
      {
        gemm_fpx(xq_head, 0, layer->wq->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
      }
      else
      {
        gemm_int8(xq_head, 0, layer->wq->i8.q + h * CH * C, 0,
            layer->wq->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
      }
      if (h >= NH_kv) continue;

      floatx *xk_head = buf->xk + h * T * CH;
      floatx *xv_head = buf->xv + h * T * CH;

      if (!quant)
      {
        // (T, NH_kv, CH)
        gemm_fpx(xk_head, 0, layer->wk->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
        gemm_fpx(xv_head, 0, layer->wv->fpx + h * CH * C, 0, buf->x, 0, T, CH,
            C, false);
      }
      else
      {
        gemm_int8(xk_head, 0, layer->wk->i8.q + h * CH * C, 0,
            layer->wk->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
        gemm_int8(xv_head, 0, layer->wv->i8.q + h * CH * C, 0,
            layer->wv->i8.scales + h * CH, buf->x_i8, 0, buf->x_scales, T, CH,
            C, false);
      }
    }
    if (g_interrupted) return 1;

    // Optional q & k norm
    if (cfg->qk_norm)
    {
      #pragma omp parallel for collapse(2)
      for (int h = 0; h < NH; h++)
      {
        for (int t = 0; t < T; t++)
        {
          // Q norm
          floatx *xq_head = buf->xq + h * T * CH + t * CH;
          rmsnorm(xq_head, xq_head, layer->nq, CH, cfg->eps);

          if (h < NH_kv)
          {
            // K norm
            floatx *xk_head = buf->xk + h * T * CH + t * CH;
            rmsnorm(xk_head, xk_head, layer->nk, CH, cfg->eps);
          }
        }
      }
    }
    if (g_interrupted) return 1;

    bool    is_local = cfg->att_layers[l];
    floatx *freqs_cs = is_local ? buf->csfreqs_slid : buf->csfreqs_full;

    // RoPE
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < NH + NH_kv; h++)
      for (int t = 0; t < T; t++)
      {
        floatx *data;
        if (h < NH)
        {
          data = buf->xq + h * T * CH + t * CH;  // Apply to queries
        }
        else
        {
          data = buf->xk + (h - NH) * T * CH + t * CH;  // Apply to keys
        }

        for (int d = 0; d < CH_half; d++)
        {
          float cfr = (float)freqs_cs[2 * d + t * CH];
          float sfr = (float)freqs_cs[2 * d + 1 + t * CH];
          float a   = (float)data[d];  // Index in the first half vector
          float b   = (float)data[d + CH_half];  // ... second half vector
          float r0  = a * cfr - b * sfr;
          float r1  = a * sfr + b * cfr;

          // Apply att_scale beforehand in this step, mathematically equivilant,
          // but avoided scaling the entire T * max_k attention matrix
          if (h < NH)
          {
            r0 *= (float)att_scale;
            r1 *= (float)att_scale;
          }

          data[d]           = (floatx)r0;
          data[d + CH_half] = (floatx)r1;
        }
      }

    if (g_interrupted) return 1;

    // (NH_kv, cache_len, CH)
    floatx *k_cache = buf->kv_cache + l * 2 * buf->cache_len * Ckv;
    floatx *v_cache = k_cache + buf->cache_len * Ckv;

    // Write to kv_cache
    for (int h = 0; h < NH_kv; h++)
    {
      floatx *xk_head = k_cache + h * buf->cache_len * CH + spos * CH;
      floatx *xv_head = v_cache + h * buf->cache_len * CH + spos * CH;
      memcpy(xk_head, buf->xk + h * T * CH, T * CH * sizeof(*buf->xk));
      memcpy(xv_head, buf->xv + h * T * CH, T * CH * sizeof(*buf->xv));
    }
    if (g_interrupted) return 1;

    int max_k = epos + 1;

    #pragma omp parallel for
    for (int h = 0; h < NH; h++)
    {
      int h_kv = h * NH_kv / NH;  // GQA mapping

      // (T, max_k)
      floatx *att_head = buf->att + h * T * max_k;  // att[h, :, :]
      // (cache_len, CH)
      floatx *xv_head = v_cache + h_kv * buf->cache_len * CH;
      floatx *xo_head = buf->xo + h * CH;  // (T, CH), column stride Cq

      /* Blockwise causal masking, the tiling algorithm used in the
       * FlashAttention paper (Dao et al., 2022,
       * https://arxiv.org/abs/2205.14135).
       *
       * The original algorithm was designed for GPUs to reduce HBM access,
       * which is not a problem for CPUs, but the idea can also be used on CPUs
       * to provide better performance for causal masking.
       *
       * Naively computing causal attention means either (a) materializing the
       * full [T, attlen] score matrix and masking out the upper triangle
       * afterwards (wastes ~half the compute), or (b) looping token by token
       * with a triangular schedule (correct FLOP count, but ragged inner loop
       * length basically kills vectorization across threads). So I tile both
       * the query and key range into blocks and classify each (qb, kb) pair
       * into five cases:
       *
       * kb > qb -> skip, every key in this block lies in the future of queries.
       * kb < qb -> full GEMM, every key lies in the past of queries.
       * kb = qb -> full GEMM & mask the upper triangle.
       * kb < spos -> skip, every key in lies before the sliding window.
       * kb = spos -> full GEMM & mask the lower triangle.
       */

      if (mask)  // Only apply tiling if mask=true
      {
        for (int q_start = 0; q_start < T; q_start += QK_BLOCK_SIZE)
        {
          // xq[h, q_start:q_end, :] (q_end - q_start, CH)
          floatx *qb          = buf->xq + h * T * CH + q_start * CH;
          int     q_end       = min(q_start + QK_BLOCK_SIZE, T);
          int     abs_q_end   = spos + q_end;
          int     abs_q_start = spos + q_start;
          if (is_local && abs_q_start >= cfg->slide_len)
          {
            abs_q_start = abs_q_start + 1 - cfg->slide_len;
          }
          else
          {
            abs_q_start = 0;
          }
          int spos_b = abs_q_start / QK_BLOCK_SIZE * QK_BLOCK_SIZE;

          for (int k_start = spos_b; k_start < abs_q_end;
               k_start += QK_BLOCK_SIZE)
          {
            // k_cache[h_kv, k_start:k_end, :] (k_end - k_start, CH)
            floatx *kb    = k_cache + h_kv * buf->cache_len * CH + k_start * CH;
            int     k_end = min(k_start + QK_BLOCK_SIZE, max_k);
            floatx *att_b = att_head + q_start * max_k + k_start;

            // Typically people don't quantize this
            gemm_fpx(att_b, max_k, kb, 0, qb, 0, q_end - q_start,
                k_end - k_start, CH, false);

            // Apply causal mask & sliding window mask
            for (int qi = q_start; qi < q_end; qi++)
            {
              int pos_i  = spos + qi;
              int spos_i = (is_local && pos_i >= cfg->slide_len)
                               ? (pos_i + 1 - cfg->slide_len)
                               : 0;
              for (int ki = k_start; ki < k_end; ki++)
              {
                if (ki > pos_i || ki < spos_i)
                {
                  att_head[qi * max_k + ki] = 0.0f;
                }
              }
            }
          }
          // Softmax & softcap for the rows
          for (int qi = q_start; qi < q_end; qi++)
          {
            floatx *att_row = att_head + qi * max_k;
            int     pos_i   = spos + qi;
            int     spos_i  = (is_local && pos_i >= cfg->slide_len)
                                  ? (pos_i + 1 - cfg->slide_len)
                                  : 0;

            // Optional tanh softcapping
            if (cfg->att_softcap != 0.0f)
            {
              for (int t = spos_i; t <= pos_i; t++)
              {
                float val  = (float)att_row[t] / cfg->att_softcap;
                att_row[t] = (floatx)(tanhf(val) * cfg->att_softcap);
              }
            }
            // only softmax in the range [spos_i, pos_i]
            softmax(att_row + spos_i, att_row + spos_i, pos_i - spos_i + 1);
          }

          // xo_block (qb_len, CH) = att_block (qb_len, qb_len)
          //                       @ v_block   (qb_len, CH)
          floatx *xo_block  = xo_head + q_start * Cq;
          floatx *att_block = att_head + q_start * max_k + spos_b;
          floatx *v_block   = xv_head + spos_b * CH;

          gemm_fpx_nn(xo_block, /*dst_stride=*/Cq, v_block, /*mat_stride=*/0,
              att_block,
              /*src_stride=*/max_k, q_end - q_start, CH, abs_q_end - spos_b,
              false);
        }
      }
      else  // mask=false, take the dense path
      {
        // (T, CH)
        floatx *xq_head = buf->xq + h * T * CH;  // xq[h, :, :]
        // k_cache[h_kv, :, :] (cache_len, CH)
        floatx *xk_head = k_cache + h_kv * buf->cache_len * CH;
        // att_head = xq_head @ xk_head.T
        gemm_fpx(att_head, max_k, xk_head, 0, xq_head, 0, T, max_k, CH, false);

        for (int qi = 0; qi < T; qi++)
        {
          floatx *att_row = att_head + qi * max_k;

          // Optional tanh softcapping
          for (int t = 0; t < max_k; t++)
          {
            if (cfg->att_softcap != 0.0f)
            {
              float val  = (float)att_row[t] / cfg->att_softcap;
              att_row[t] = (floatx)(tanhf(val) * cfg->att_softcap);
            }
          }
          // Softmax
          softmax(att_row, att_row, max_k);
        }
        // xo_head (T, CH) = att_head (T, max_k) @ xv_head[:max_k] (max_k, CH)
        gemm_fpx_nn(xo_head, /*dst_stride=*/Cq, xv_head, /*mat_stride=*/0,
            att_head, /*src_stride=*/max_k, T, CH, max_k, false);
      }
    }
    if (g_interrupted) return 1;
    // x (T, C) = xo_head (T, CH) @ wo.T (CH, C)
    if (!quant)
    {
      gemm_fpx(buf->x, 0, layer->wo->fpx, 0, buf->xo, Cq, T, C, Cq, true);
    }
    else
    {
      quantize_acts(buf->xo_i8, buf->xo_scales, buf->xo, Cq, T, Cq, true);
      gemm_int8(buf->x, 0, layer->wo->i8.q, 0, layer->wo->i8.scales, buf->xo_i8,
          0, buf->xo_scales, T, C, Cq, true);
    }
    if (g_interrupted) return 1;

    #pragma omp parallel for
    for (int t = 0; t < T; t++)
    {
      floatx *x_row = buf->x + t * C;
      rmsnorm(x_row, x_row, layer->n2, C, cfg->eps);
    }
    if (g_interrupted) return 1;

    floatx *restrict x     = buf->x;
    floatx *restrict resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Combine the residual stream
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;

    memcpy(buf->resid, buf->x, T * C * sizeof(*buf->x));

    if (cfg->pre_mlp_norm)
    {
      #pragma omp parallel for
      for (int t = 0; t < T; t++)
      {
        floatx *x_row = buf->x + t * C;
        rmsnorm(x_row, x_row, layer->n3, C, cfg->eps);
      }
    }
    if (g_interrupted) return 1;

    // MLP
    if (!quant)
    {
      gemm_fpx(
          buf->xu, 0, layer->w1->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true);
      gemm_fpx(
          buf->xg, 0, layer->w2->fpx, 0, buf->x, 0, T, cfg->mlp_dim, C, true);
    }
    else
    {
      quantize_acts(buf->x_i8, buf->x_scales, buf->x, 0, T, C, true);
      gemm_int8(buf->xu, 0, layer->w1->i8.q, 0, layer->w1->i8.scales, buf->x_i8,
          0, buf->x_scales, T, cfg->mlp_dim, C, true);
      gemm_int8(buf->xg, 0, layer->w2->i8.q, 0, layer->w2->i8.scales, buf->x_i8,
          0, buf->x_scales, T, cfg->mlp_dim, C, true);
    }
    if (g_interrupted) return 1;

    // GELU gate
    #pragma omp parallel for
    for (int d = 0; d < T * cfg->mlp_dim; d++)
    {
      // Tanh approximation of GELU
      float x    = (float)buf->xg[d];
      float c    = 0.79788456080287f;  // = sqrt(2 / pi)
      x          = 0.5 * x * (1 + tanhf(c * (x + 0.044715 * x * x * x)));
      buf->xg[d] = (floatx)x;
      buf->xg[d] *= buf->xu[d];  // Fuse xg * xu into xg
    }
    if (g_interrupted) return 1;

    // Down projection
    if (!quant)
    {
      gemm_fpx(
          buf->x, 0, layer->w3->fpx, 0, buf->xg, 0, T, C, cfg->mlp_dim, true);
    }
    else
    {
      quantize_acts(
          buf->xg_i8, buf->xg_scales, buf->xg, 0, T, cfg->mlp_dim, true);
      gemm_int8(buf->x, 0, layer->w3->i8.q, 0, layer->w3->i8.scales, buf->xg_i8,
          0, buf->xg_scales, T, C, cfg->mlp_dim, true);
    }
    if (g_interrupted) return 1;

    if (cfg->pst_mlp_norm)
    {
      #pragma omp parallel for
      for (int t = 0; t < T; t++)
      {
        floatx *x_row = buf->x + t * C;
        rmsnorm(x_row, x_row, layer->n4, C, cfg->eps);
      }
    }
    if (g_interrupted) return 1;

    x     = buf->x;
    resid = buf->resid;
    #pragma omp parallel for
    for (int d = 0; d < T * C; d++)
    {
      // Second residual
      buf->x[d] = clamp_fpx(x[d] + resid[d]);
    }
    if (g_interrupted) return 1;
  }

  // Final RMSNorm
  #pragma omp parallel for
  for (int t = 0; t < T; t++)
  {
    floatx *x_row = buf->x + t * C;
    rmsnorm(x_row, x_row, dec->final_norm, C, cfg->eps);
  }

  // Compute logits (tied embedding)
  if (compute_logits)
  {
    floatx *last_x = buf->x + (T - 1) * C;
    if (!quant)
    {
      gemv_fpx(
          buf->logits, dec->embedding->fpx, last_x, cfg->vocab_size, C, true);
    }
    else
    {
      floatx xscale = quantize_act(buf->x_i8, last_x, C);
      gemv_int8(buf->logits, dec->embedding->i8.q, dec->embedding->i8.scales,
          buf->x_i8, xscale, cfg->vocab_size, C, true);
    }
    if (g_interrupted) return 1;

    // Optional logit softcapping
    if (cfg->logit_softcap != 0.0f)
    {
      for (int d = 0; d < cfg->vocab_size; d++)
      {
        float val      = (float)buf->logits[d] / cfg->logit_softcap;
        buf->logits[d] = (floatx)(tanhf(val) * cfg->logit_softcap);
      }
    }
    if (g_interrupted) return 1;
  }
  return 0;
}

/* Language + vision model forward */
int
forward_gemma_decode(GemmaModel *model, TextBuffer *buf, int token, int pos)
{
  TextDecoder *dec = model->decoder;

  /* Embedding lookup.
   * The export script already applied 1/sqrt(embed_dim), so the usual
   * Gemma sqrt(embed_dim) scale cancels out to 1.0 here. */
  floatx embed_scale =
      1.0f;  // equivalent to sqrt(embed_dim) * (1/sqrt(embed_dim))
  if (model->quant)
  {
    // Dequantize
    embed_scale *= dec->embedding->i8.scales[token];
  }

  int C = dec->config->embed_dim;

  // x = embedding[tok] * embed_scale
  #pragma omp parallel for
  for (int d = 0; d < C; d++)
  {
    if (!model->quant)
    {
      buf->x[d] = dec->embedding->fpx[token * C + d] * embed_scale;
    }
    else
    {
      buf->x[d] = (floatx)dec->embedding->i8.q[token * C + d] * embed_scale;
    }
  }
  if (g_interrupted) return 1;
  return forward_text_decode(model->decoder, buf, pos, model->quant, true);
}

/* Language model prefill */
int
forward_gemma_prefill(GemmaModel *model,
    TextBuffer                   *buf,

    int  *tokens,
    int  *pos,
    int   chunk_size,
    bool *rpen_visited,
    bool  compute_logits)
{
  TextDecoder *dec = model->decoder;
  int          T   = 0;
  for (int *t = tokens; *t != EOF; t++)
  {
    if (rpen_visited != NULL)
    {
      rpen_visited[*t] = true;
    }
    T++;
  }

  int C = dec->config->embed_dim;

  if (T <= PREFILL_FALLBACK_THRESHOLD)
  {
    // Decode every token
    for (int i = 0; i < T; i++)
    {
      int token = tokens[i];

      floatx embed_scale = 1.0f;
      if (model->quant)
      {
        embed_scale *= dec->embedding->i8.scales[token];
      }

      #pragma omp parallel for
      for (int d = 0; d < C; d++)
      {
        if (!model->quant)
          buf->x[d] = dec->embedding->fpx[token * C + d] * embed_scale;
        else
          buf->x[d] = (floatx)dec->embedding->i8.q[token * C + d] * embed_scale;
      }

      bool need_logits = (i == T - 1) && compute_logits;

      if (forward_text_decode(dec, buf, *pos + i, model->quant, need_logits) !=
          0)
        return 1;
    }
  }
  else
  {
    // Prefill by chunks
    for (int off = 0; off < T; off += chunk_size)
    {
      int  cur_len       = min(chunk_size, T - off);
      bool is_last_chunk = (off + cur_len == T);

      // Embedding lookup
      for (int t = 0; t < cur_len; t++)
      {
        int token   = tokens[off + t];
        int emb_off = t * C;

        for (int d = 0; d < C; d++)
        {
          if (!model->quant)
          {
            buf->x[emb_off + d] = dec->embedding->fpx[token * C + d];  // * 1.0f
          }
          else
          {
            buf->x[emb_off + d] = (floatx)dec->embedding->i8.q[token * C + d] *
                                  dec->embedding->i8.scales[token];
          }
        }
      }

      int rc = forward_text_chunk(dec, buf, *pos + off, cur_len, true,
          model->quant, is_last_chunk && compute_logits);

      if (rc != 0) return rc;
    }
  }

  *pos += T;
  return 0;
}

/* Vision model forward + embedding prefill */
int
forward_gemma_image(GemmaModel *model,
    TextBuffer                 *buf,
    VisionBuffer               *vbuf,

    const floatx *image,
    int          *pos,
    bool          compute_logits)
{
  VisionEncoder *enc = model->encoder;
  if (enc == NULL)
  {
    return 1;
  }

  TextDecoder *dec = model->decoder;
  if (forward_vision(enc, dec->config, buf, vbuf, image, model->quant) == 1)
  {
    return 1;
  }

  int image_toks = dec->config->image_toks;

  int suc = forward_text_chunk(
      // Gemma 3 models uses bi-directional attention for vision tokens
      dec, buf, *pos, image_toks, false, model->quant, compute_logits);

  *pos += image_toks;
  return suc;
}

// Sampling

/* */
static int
argmax(floatx *logits, int vocab_size)
{
  // Pick the index with the max value
  int    max_idx = -1;
  floatx max_val = -(floatx)INFINITY;
  #pragma omp parallel
  {
    int    local_idx = -1;
    floatx local_val = -(floatx)INFINITY;
    #pragma omp for nowait
    for (int i = 0; i < vocab_size; i++)
    {
      if (logits[i] > local_val)
      {
        local_val = logits[i];
        local_idx = i;
      }
    }
    #pragma omp critical
    {
      // Only one thread is able to run this at a time
      if (local_val > max_val)
      {
        max_val = local_val;
        max_idx = local_idx;
      }
    }
  }
  return max_idx;
}

/* */
typedef struct
{
  floatx val;
  int    idx;
} FloatIdx;

/* */
static inline void
swap_fi(FloatIdx *a, FloatIdx *b)
{
  FloatIdx t = *a;

  *a = *b;
  *b = t;
}

/* Tiny xorshift for pivot selection */
static uint32_t qs_rand_state = 3418323524;
static inline uint32_t
qs_rand(void)
{
  qs_rand_state ^= qs_rand_state << 13;
  qs_rand_state ^= qs_rand_state >> 7;
  qs_rand_state ^= qs_rand_state << 17;
  return (uint32_t)qs_rand_state;
}

/* */
static int
partition_desc(FloatIdx *arr, int lo, int hi)
{
  // Pick the pivot randomly (use a seperate rand sequence)
  int r = (int)lo + (int)qs_rand() % (hi - lo + 1);
  swap_fi(&arr[r], &arr[hi]);

  floatx pivot = arr[hi].val;

  int i = lo;
  for (int j = lo; j < hi; j++)
  {
    if (arr[j].val > pivot)
    {
      // Put the greater one on the left
      swap_fi(&arr[i++], &arr[j]);
    }
  }
  swap_fi(&arr[i], &arr[hi]);
  return i;
}

/* Quickselect to find the top-k elements (descending) */
static void
quickselect_topk(FloatIdx *arr, int lo, int hi, int k_idx)
{
  while (lo < hi)
  {
    int p = partition_desc(arr, lo, hi);
    if (p == k_idx) return;
    else if (p < k_idx)
    {
      lo = p + 1;
    }
    else
    {
      hi = p - 1;
    }
  }
}

/* */
static void
apply_topk(floatx *logits, FloatIdx *logit_indices, int vocab_size, int k)
{
  if (k <= 0)
  {
    k = 1;
  }
  if (k > vocab_size)
  {
    k = vocab_size;
  }

  // Record index info
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logit_indices[i].idx = i;
    logit_indices[i].val = logits[i];
  }
  quickselect_topk(logit_indices, 0, vocab_size - 1, k - 1);

  // Keep the top k channels
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logits[i] = -(floatx)INFINITY;
  }
  for (int i = 0; i < k; i++)
  {
    logits[logit_indices[i].idx] = logit_indices[i].val;
  }
}

/* Max-heap helpers for top-p */
static void
sift_down(FloatIdx *arr, int n, int i)
{
  // Make sure the parent node arr[i] is greater than its children in
  // the heap
  for (;;)
  {
    // l & r are the two children node
    int l = 2 * i + 1, r = 2 * i + 2, largest = i;
    if (l < n && arr[l].val > arr[largest].val) largest = l;
    if (r < n && arr[r].val > arr[largest].val) largest = r;
    if (largest == i) break;
    swap_fi(&arr[i], &arr[largest]);
    i = largest;
  }
}

/* */
static void
build_heap(FloatIdx *arr, int n)
{
  for (int i = n / 2 - 1; i >= 0; i--)
  {
    sift_down(arr, n, i);
  }
}

/* */
static void
apply_topp(floatx *logits,
    floatx        *fpbuf,
    FloatIdx      *logit_indices,

    int   vocab_size,
    int   k,
    float p)
{
  if (k > vocab_size)
  {
    k = vocab_size;
  }
  // Softmax to get the probs, store in fpbuf
  softmax(fpbuf, logits, vocab_size);
  int heap_size = (k == 0) ? vocab_size : k;

  if (k == 0)
  {
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; i++)
    {
      logit_indices[i].idx = i;
      logit_indices[i].val = fpbuf[i];
    }
  }
  else
  {
    for (int i = 0; i < k; i++)
    {
      int idx = logit_indices[i].idx;  // Reuse the candidates from topk
      logit_indices[i].val = fpbuf[idx];
    }
  }
  build_heap(logit_indices, heap_size);  // O(k)

  // fpbuf is now a copy of the original logits
  memcpy(fpbuf, logits, vocab_size * sizeof(floatx));

  // Set logits to -inf
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    logits[i] = -(floatx)INFINITY;
  }

  float cum = 0.0f;  // Cumulative prob

  while (heap_size > 0)
  {
    // Pop the current max prob
    FloatIdx top     = logit_indices[0];
    logit_indices[0] = logit_indices[--heap_size];  // Put the last element to
                                                    // the top
    sift_down(logit_indices, heap_size, 0);         // O(log(vocab_size))

    logits[top.idx] = fpbuf[top.idx];
    cum += (float)top.val;
    if (cum >= p) break;
  }
}

/* Simple repetition penalty */
void
apply_rpen(floatx *logits, bool *visited, int vocab_size, float rpen)
{
  // rpen short for Repetition Penalty
  #pragma omp parallel for
  for (int i = 0; i < vocab_size; i++)
  {
    if (!visited[i]) continue;
    float val = (float)logits[i];
    if (val > 0.0f)
    {
      logits[i] = (floatx)(val / rpen);
    }
    else
    {
      logits[i] = (floatx)(val * rpen);
    }
  }
}

/* Sample the next token from the logits */
int
sample_from_logits(floatx *logits,
    floatx                *probs,
    FloatIdx              *logit_indices,
    bool                  *visited,
    int                    vocab_size,
    float                  temperature,
    int                    topk,
    float                  topp,
    float                  rpen)
{
  bool dosample = temperature != 0 && topk != 1;

  // Manipulate the logits & sample the next token
  if (!dosample)
  {
    // Argmax sampling
    return argmax(logits, vocab_size);
  }

  bool use_topk = dosample && topk != 0;
  bool use_topp = dosample && topp < 1.0f;
  bool use_rpen = dosample && rpen > 1.0f;

  // Apply the temperature
  #pragma omp parallel for
  for (int d = 0; d < vocab_size; d++)
  {
    logits[d] /= (floatx)temperature;
  }

  if (use_topk)
  {
    apply_topk(logits, logit_indices, vocab_size, topk);
  }
  if (use_topp)
  {
    apply_topp(logits, probs, logit_indices, vocab_size, topk, topp);
  }
  if (use_rpen)
  {
    apply_rpen(logits, visited, vocab_size, rpen);
  }

  // Softmax to get the probs
  softmax(probs, logits, vocab_size);

  // Multinomial sample from probs
  float r   = (float)((float)rand() / (RAND_MAX + 1.0));
  float sum = 0.0f;

  int token = vocab_size - 1;
  for (int d = 0; d < vocab_size; d++)
  {
    sum += (float)probs[d];
    if (r < sum)
    {
      token = d;
      break;
    }
  }
  return token;
}

typedef enum
{
  INJECT_NONE,
  INJECT_TEXT,
  INJECT_IMAG
} InjectType;

typedef struct
{
  InjectType type;
  bool       prefill_finished;
  bool       quit;
  union
  {
    int    *tokens;
    floatx *image;
  };
} InjectData;

/* The main sampling loop */
void
sample(GemmaModel *model,
    TextBuffer    *buf,
    VisionBuffer  *vbuf,
    InjectData     init_data,

    int   seqlen,
    int   chunk_size,
    float temperature,
    int   topk,
    float topp,
    float rpen,
    bool  enable_mm,

    void *cb_ctx,
    InjectData (*token_callback)(
        int token, GemmaModel *model, bool enable_mm, void *ctx))
{
  TextConfig *cfg = model->decoder->config;
  int         vs  = cfg->vocab_size;

  // Boolean flags
  bool dosample = temperature != 0 && topk != 1;
  bool use_topk = dosample && topk != 0;
  bool use_topp = dosample && topp < 1.0f;
  bool use_rpen = dosample && rpen > 1.0f;

  double prefill_start   = 0.0f;
  double prefill_end     = 0.0f;
  double prefill_elapsed = 0.0f;
  double gen_start       = 0.0f;
  double gen_end         = 0.0f;
  double gen_elapsed     = 0.0f;

  int     prompt_toks = 0;
  int     gen_toks    = 0;
  int     pos         = 0;
  int     token       = 0;
  floatx *probs       = NULL;

  // Injected data (text chunk / image) produced by the callback
  InjectData injected = init_data;
  // Temporary buffers for top‑k / top‑p sorting
  FloatIdx *logit_indices = NULL;

  // visited[] tracks which token IDs have been seen, used for repetition
  // penalty
  bool *visited = NULL;
  if (use_rpen)
  {
    CALLOC(visited, vs, "visited", goto end;);
  }

  // Only allocate probs if we need to sample
  if (dosample)
  {
    MALLOC(probs, vs, "probs", goto end;);
  }

  if (use_topk || use_topp)
  {
    MALLOC(logit_indices, vs, "logit_indices", goto end;);
  }

  // Record tok/s
  gen_toks = 0;

  // Whether we have finished prefilling the initial prompt
  // Initially set by the first InjectData from the callback
  bool prefill_finished = init_data.prefill_finished;

  while (pos < seqlen && !injected.quit)
  {
    // Case A: Normal auto‑regressive generation
    if (injected.type == INJECT_NONE)
    {
      gen_start = now_sec();
      // Run one decoding step, given the current token, produce logits
      if (forward_gemma_decode(model, buf, token, pos) == 1) goto end;
      prefill_finished = true;  // We are now in generation phase
      pos++;
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;
    }
    else  // Case B: Data injection (prefill or image)
    {
      int prev_pos     = pos;
      prefill_start    = now_sec();
      prefill_finished = injected.prefill_finished;

      if (injected.type == INJECT_TEXT)
      {
        // Prefill a chunk of text tokens (ends with EOF sentinel)
        if (forward_gemma_prefill(model, buf, injected.tokens, &pos, chunk_size,
                visited, prefill_finished) == 1)
          goto end;
      }
      else if (injected.type == INJECT_IMAG)
      {
        // Inject an image, run the vision encoder and prefill its soft tokens
        if (forward_gemma_image(
                model, buf, vbuf, injected.image, &pos, prefill_finished) == 1)
          goto end;
        free(injected.image);
      }

      prefill_end = now_sec();
      prefill_elapsed += prefill_end - prefill_start;
      prompt_toks += pos - prev_pos;
    }
    if (g_interrupted) break;

    if (prefill_finished)
    {
      gen_start = now_sec();

      // Manipulate the logits & sample the next token
      if (g_interrupted) break;
      token = sample_from_logits(buf->logits, probs, logit_indices, visited, vs,
          temperature, topk, topp, rpen);
      gen_toks++;
      gen_end = now_sec();
      gen_elapsed += gen_end - gen_start;

      injected = token_callback(token, model, enable_mm, cb_ctx);
    }
    else
    {
      // Injection not complete, send EOF to request the next chunk of data
      injected = token_callback(EOF, model, enable_mm, cb_ctx);
    }
  }

end:
  // Print prefilling speed
  if (prefill_elapsed > 0.0)
  {
    printf("\nPrompt processed %d tokens in %.2f seconds (%.2f tok/s)\n",
        prompt_toks, prefill_elapsed, prompt_toks / prefill_elapsed);
  }
  else
  {
    printf("\nPrompt processed %d tokens instantly\n", prompt_toks);
  }
  // Print generation speed
  if (gen_elapsed > 0.0)
  {
    printf("Generated %d tokens in %.2f seconds (%.2f tok/s)\n", gen_toks,
        gen_elapsed, gen_toks / gen_elapsed);
  }
  else
  {
    printf("Generated %d tokens instantly\n", gen_toks);
  }

  free(probs);
  if (use_rpen)
  {
    free(visited);
  }
  if (use_topk || use_topp)
  {
    free(logit_indices);
  }
}

/* Shared "@image{path}" scanning state, used by both generate and chat */
typedef struct
{
  // Scratch token buffer (caller-owned) that inject_next_chunk() encodes into.
  // Sized generously (seqlen + slack for template/special tokens)
  int *tokens_buf;
  int  tokens_buf_len;
  int  token_start;  // Start of the slice returned by RETURN_TEXT_INJECT
  int  n_tokens;     // Write cursor / total tokens encoded so far

  // Remaining un-encoded prompt text (NUL-terminated). Advances as chunks are
  // consumed; owned by the caller (points into its own buffer)
  const char *text;

  // True right after an "@image{path}" is matched: the next call should load &
  // inject the image itself
  bool pending_image;
  // True right after an image was injected: the next call should close it off
  // with <end_of_image>
  bool was_image;
  char pending_path[4096];
} PromptCursor;

// clang-format off
// Seal off pc->tokens_buf[pc->token_start : pc->n_tokens] into an InjectData
// slice, advance token_start, and return it
#define RETURN_TEXT_INJECT(pc, prefill_finished_val, quit_val) \
  do                                                           \
  {                                                            \
    int start_                       = (pc)->token_start;      \
    (pc)->token_start                = (pc)->n_tokens;         \
    (pc)->tokens_buf[(pc)->n_tokens] = EOF;                    \
    return (InjectData){.type = INJECT_TEXT,                   \
        .tokens               = (pc)->tokens_buf + start_,     \
        .prefill_finished     = (prefill_finished_val),        \
        .quit                 = (quit_val)};                   \
  }                                                            \
  while (0)
// clang-format on

/* Scan pc->text for the next "@image{path}", one step at a time:
 * if an image path was queued last call, load it and inject it
 * if an image was just injected, close it off with <end_of_image>
 * if a complete "@image{path}" is found, encode any leading text (if any) and
 *     queue the path for the next call
 * otherwise (no more complete "@image{...}" in pc->text), leave pc->text
 *     untouched, set *done = true, and let the caller decide how to consume/
 *     finish the remaining plain text (this differs between generate()'s raw
 *     prompt and chat()'s templated turn)
 */
static InjectData
inject_next_chunk(GemmaTokenizer *tok,
    GemmaModel                   *model,
    bool                          enable_mm,
    PromptCursor                 *pc,
    bool                         *done)
{
  *done = false;

  VisionEncoder *enc = model->encoder;

  if (pc->pending_image)
  {
    pc->pending_image = false;
    // Load the saved image path
    int     img_size = (enc && enable_mm) ? enc->config->image_size : 0;
    floatx *img      = prepare_image(pc->pending_path, img_size);
    if (img == NULL)
    {
      fprintf(stderr, "error: failed to load image: '%s'\n", pc->pending_path);
      return (InjectData){.type = INJECT_NONE, .quit = true};
    }
    pc->was_image = true;
    return (InjectData){.type = INJECT_IMAG,
        .image                = img,
        .prefill_finished     = false,
        .quit                 = false};
  }

  if (pc->was_image)
  {
    pc->was_image                  = false;
    pc->tokens_buf[pc->n_tokens++] = tok->eoi;  // <end_of_image>
    encode(tok, "\n\n", 2, pc->tokens_buf + pc->n_tokens, &pc->n_tokens);
  }

  // Use @image{<path>} to insert an image
  const char *prompt    = pc->text;
  char       *image_cmd = strstr(prompt, "@image{");
  if (enc == NULL || !enable_mm)
  {
    image_cmd = NULL;
  }

  char *closing = (image_cmd != NULL) ? strchr(image_cmd, (int)'}') : NULL;
  if (image_cmd == NULL || closing == NULL)
  {
    // No (more) complete "@image{...}" in the remaining text -- nothing
    // left for this function to do; the caller consumes pc->text itself
    *done = true;
    return (InjectData){.type = INJECT_NONE, .quit = false};
  }

  int spos = (int)(image_cmd - prompt);
  int epos = (int)(closing - prompt);

  if (spos == 0)
  {
    // Image at the start of chunk
    const char *path     = image_cmd + strlen("@image{");
    int         path_len = epos - (int)strlen("@image{");
    if (path_len < 0 || (size_t)path_len >= sizeof(pc->pending_path))
    {
      fprintf(stderr, "error: invalid image path\n");
      return (InjectData){.type = INJECT_NONE, .quit = true};
    }
    memcpy(pc->pending_path, path, path_len);
    pc->pending_path[path_len] = '\0';
    pc->pending_image          = true;

    encode(tok, "\n\n", 2, pc->tokens_buf + pc->n_tokens, &pc->n_tokens);
    pc->tokens_buf[pc->n_tokens++] = tok->soi;  // Insert <start_of_image>
    pc->text                       = prompt + epos + 1;
    RETURN_TEXT_INJECT(pc, false, false);
  }
  // Matched a complete image command! Encode & return the text part first,
  // handle the image part in the next iteration
  encode(tok, prompt, spos, pc->tokens_buf + pc->n_tokens, &pc->n_tokens);
  // No need to insert <start_of_image> here, leave it to the next iteration
  pc->text = prompt + spos;  // Next call directly starts with "@image"
  RETURN_TEXT_INJECT(pc, false, false);
}

/* */
static InjectData
generate_next(GemmaModel *model, bool enable_mm, PromptCursor *pc)
{
  GemmaTokenizer *tok = model->decoder->tokenizer;

  bool       done;
  InjectData r = inject_next_chunk(tok, model, enable_mm, pc, &done);
  if (!done) return r;
  if (r.quit) return r;

  // No more "@image{...}" ahead, generate() has no chat template to
  // append, so just encode whatever plain text remains and finish up
  if (pc->text[0] != '\0')
  {
    encode(tok, pc->text, strlen(pc->text), pc->tokens_buf + pc->n_tokens,
        &pc->n_tokens);
    pc->text += strlen(pc->text);
    RETURN_TEXT_INJECT(pc, true, false);
  }
  r.prefill_finished = true;
  return r;
}

/* */
static InjectData
generate_callback(int token, GemmaModel *model, bool enable_mm, void *ctx)
{
  PromptCursor   *pc  = (PromptCursor *)ctx;
  GemmaTokenizer *tok = model->decoder->tokenizer;

  if (token == EOF)
  {
    // EOF: continue feeding the prompt (text / image chunks)
    return generate_next(model, enable_mm, pc);
  }
  if (token == tok->eos || token == tok->eot)
  {
    return (InjectData){.type = INJECT_NONE, .quit = true};
  }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return (InjectData){.type = INJECT_NONE, .quit = false};
}

/* */
int
generate(GemmaModel *model,
    TextBuffer      *buf,
    VisionBuffer    *vbuf,

    const char *prompt,
    int         seqlen,
    int         chunk_size,
    float       temperature,
    int         topk,
    float       topp,
    float       rpen,
    bool        enable_mm)
{
  GemmaTokenizer *tok = model->decoder->tokenizer;

  printf("%s", prompt);

  int  tokens_buf_len = seqlen + 128;
  int *tokens_buf;
  MALLOC(tokens_buf, tokens_buf_len, "tokens_buf", return 1;);

  PromptCursor pc = {
      .tokens_buf     = tokens_buf,
      .tokens_buf_len = tokens_buf_len,
      .token_start    = 0,
      .n_tokens       = 0,
      .text           = prompt,
      .pending_image  = false,
      .was_image      = false,
  };
  pc.tokens_buf[pc.n_tokens++] = tok->bos;

  InjectData first = generate_next(model, enable_mm, &pc);
  if (first.quit)
  {
    free(tokens_buf);
    return 1;
  }

  sample(model, buf, vbuf, first, seqlen, chunk_size, temperature, topk, topp,
      rpen, enable_mm, &pc, generate_callback);

  free(tokens_buf);
  return 0;
}

/* Per-chat-session state: wraps a PromptCursor (for "@image{...}" parsing)
 * plus the bits specific to reading turns from stdin and wrapping them in
 * the Gemma chat template. */
typedef struct
{
  PromptCursor pc;
  bool         turn_finished;
  char        *line_buf;  // One line of stdin input for the current turn
  int          line_buf_len;
} ChatState;

/* Build the Gemma chat template */
static InjectData
new_turn(ChatState *cs,
    GemmaTokenizer *tok,
    GemmaModel     *model,
    bool            enable_mm,
    bool            bos)
{
  /* Template (from https://ai.google.dev/gemma/docs/core/prompt-structure):
   * <start_of_turn>user\n
   * What is Cramer's Rule?<end_of_turn>\n
   * <start_of_turn>model */

  PromptCursor *pc = &cs->pc;

  if (cs->turn_finished)
  {
    // Get a fresh input from stdin
    if (bos)
    {
      printf("User: ");
    }
    else
    {
      printf("\nUser: ");
    }

    if (fgets(cs->line_buf, cs->line_buf_len, stdin) == NULL)
    {
      if (!g_interrupted)
      {
        fprintf(stderr, "\nerror: failed to read user input\n");
      }
      goto fail;
    }
    if (strchr(cs->line_buf, '\n') == NULL)
    {
      // Input is truncated (too long)
      fprintf(stderr, "\nerror: input too long\n");
      goto fail;
    }
    if (g_interrupted)
    {
      goto fail;
    }
    // Set string end
    cs->line_buf[strcspn(cs->line_buf, "\n")] = '\0';

    cs->turn_finished = false;
    pc->text          = cs->line_buf;
    pc->token_start   = 0;
    pc->n_tokens      = 0;

    if (bos)
    {
      pc->tokens_buf[pc->n_tokens++] = tok->bos;
    }
    pc->tokens_buf[pc->n_tokens++] = tok->sot;
    encode(tok, "user\n", strlen("user\n"), pc->tokens_buf + pc->n_tokens,
        &pc->n_tokens);
    printf("Model: ");
  }

  bool       done;
  InjectData r = inject_next_chunk(tok, model, enable_mm, pc, &done);
  if (!done) return r;
  if (r.quit) goto fail;

  // No more "@image{...}" ahead, encode whatever plain text remains and
  // wrap up the turn with the chat template's closing tokens
  if (pc->text[0] != '\0')
  {
    encode(tok, pc->text, strlen(pc->text), pc->tokens_buf + pc->n_tokens,
        &pc->n_tokens);
  }
  cs->turn_finished              = true;
  pc->tokens_buf[pc->n_tokens++] = tok->eot;
  pc->tokens_buf[pc->n_tokens++] = get_token_idx(tok, "\n");
  pc->tokens_buf[pc->n_tokens++] = tok->sot;
  encode(tok, "model\n", strlen("model\n"), pc->tokens_buf + pc->n_tokens,
      &pc->n_tokens);
  RETURN_TEXT_INJECT(pc, true, false);

fail:
  return (InjectData){.type = INJECT_NONE, .quit = true};
}

/* */
static InjectData
chat_callback(int token, GemmaModel *model, bool enable_mm, void *ctx)
{
  ChatState      *cs  = (ChatState *)ctx;
  TextDecoder    *dec = model->decoder;
  GemmaTokenizer *tok = dec->tokenizer;

  if (token == EOF || token == tok->eos || token == tok->eot)
  {
    // EOF, or end of turn: continue/start the injection process
    return new_turn(cs, tok, model, enable_mm, false);
  }
  char byte_buf[2];
  printf("%s", decode(tok, token, byte_buf));
  fflush(stdout);
  return (InjectData){.type = INJECT_NONE, .quit = false};
}

/* */
int
chat(GemmaModel  *model,
    TextBuffer   *buf,
    VisionBuffer *vbuf,

    int   seqlen,
    int   chunk_size,
    float temperature,
    int   topk,
    float topp,
    float rpen,
    bool  enable_mm)
{
  GemmaTokenizer *tok = model->decoder->tokenizer;

  int  tokens_buf_len = seqlen + 128;
  int *tokens_buf;
  MALLOC(tokens_buf, tokens_buf_len, "tokens_buf", return 1;);

  int   line_buf_len = seqlen;
  char *line_buf;
  MALLOC(line_buf, line_buf_len, "line_buf", free(tokens_buf); return 1;);

  ChatState cs = {
      .pc = {.tokens_buf = tokens_buf, .tokens_buf_len = tokens_buf_len},
      .turn_finished = true,
      .line_buf      = line_buf,
      .line_buf_len  = line_buf_len,
  };

  InjectData first_turn = new_turn(&cs, tok, model, enable_mm, true);
  if (first_turn.quit)
  {
    free(tokens_buf);
    free(line_buf);
    return 1;
  }
  sample(model, buf, vbuf, first_turn, seqlen, chunk_size, temperature, topk,
      topp, rpen, enable_mm, &cs, chat_callback);

  free(tokens_buf);
  free(line_buf);
  return 0;
}

// CLI

/* */
static inline bool
safe_atoui(const char *str, unsigned int *result)
{
  if (str == NULL) return false;

  errno = 0;

  char     *endptr = NULL;
  long long val    = strtoll(str, &endptr, 10);

  // Invalid number
  if (endptr == str) return false;
  // Extra characters at the end
  if (*endptr != '\0') return false;
  // long overflow
  if (errno == ERANGE) return false;
  // uint overflow
  if (val < 0 || val > (long long)UINT_MAX) return false;

  // All passed
  *result = (unsigned int)val;
  return true;
}

/* */
static inline bool
safe_atof(const char *str, float *result)
{
  if (str == NULL) return false;

  errno = 0;

  char *endptr = NULL;
  float val    = strtof(str, &endptr);

  // Invalid number
  if (endptr == str) return false;
  // Extra characters at the end
  if (*endptr != '\0') return false;
  // Overflow
  if (errno == ERANGE) return false;
  // inf / nan
  if (isinf(val) || isnan(val)) return false;

  // All passed
  *result = val;
  return true;
}

#ifdef _WIN32
#  include <windows.h>
// The default console encoding is kinda weird on Windows
/* */
static void
set_utf8_console(void)
{
  SetConsoleOutputCP(65001);
  SetConsoleCP(65001);
}

/* Convert Windows command line to UTF-8 argc/argv */
static char **
get_utf8_argv(int *argc_out)
{
  wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), argc_out);
  if (!wargv) return NULL;

  char **argv = malloc((*argc_out + 1) * sizeof(char *));
  if (!argv)
  {
    LocalFree(wargv);
    return NULL;
  }

  for (int i = 0; i < *argc_out; i++)
  {
    int size =
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, NULL, 0, NULL, NULL);
    argv[i] = malloc(size);
    if (!argv[i])
    {
      for (int j = 0; j < i; j++)
      {
        free(argv[j]);
      }

      free(argv);
      LocalFree(wargv);
      return NULL;
    }
    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, argv[i], size, NULL, NULL);
  }
  argv[*argc_out] = NULL;

  LocalFree(wargv);
  return argv;
}

/* */
static void
free_utf8_argv(char **argv, int argc)
{
  if (argv == NULL) return;
  for (int i = 0; i < argc; i++)
  {
    free(argv[i]);
  }
  free(argv);
}

#else
// No problem with POSIX though
/* */
static void
set_utf8_console(void)
{
}

/* */
static char **
get_utf8_argv(int *argc_out)
{
  (void)argc_out;
  return NULL;
}

/* */
static void
free_utf8_argv(char **argv, int argc)
{
  (void)argv;
  (void)argc;
}
#endif

/* Pretty-print of the loaded model */
void
print_model_config(GemmaModel *model, int seqlen, bool enable_mm)
{
  const int       width = 20;
  VisionEncoder  *enc   = model->encoder;
  TextDecoder    *dec   = model->decoder;
  TextConfig     *cfg   = dec->config;
  GemmaTokenizer *tok   = dec->tokenizer;

  printf("\n========== Model Configuration ==========\n");
  printf("Architecture:\n");

  // Integer fields
  printf("  %-*s: %d\n", width, "n_layers", cfg->n_layers);
  printf("  %-*s: %d\n", width, "n_heads", cfg->n_heads);
  printf("  %-*s: %d\n", width, "n_kv_heads", cfg->n_kv_heads);
  printf("  %-*s: %d\n", width, "head_dim", cfg->head_dim);
  printf("  %-*s: %d\n", width, "embed_dim", cfg->embed_dim);
  printf("  %-*s: %d\n", width, "mlp_dim", cfg->mlp_dim);
  printf("  %-*s: %d\n", width, "q_scale", cfg->q_scale);
  printf("  %-*s: %d\n", width, "slide_len", cfg->slide_len);
  printf("  %-*s: %d\n", width, "image_toks", cfg->image_toks);
  printf("  %-*s: %d\n", width, "max_seqlen", cfg->max_seqlen);
  printf("  %-*s: %d\n", width, "vocab_size", cfg->vocab_size);

  // Float fields
  printf("  %-*s: %.6f\n", width, "local_theta", cfg->local_theta);
  printf("  %-*s: %.6f\n", width, "global_theta", cfg->global_theta);
  printf("  %-*s: %.6f\n", width, "eps", cfg->eps);
  printf("  %-*s: %.6f\n", width, "att_softcap", cfg->att_softcap);
  printf("  %-*s: %.6f\n", width, "logit_softcap", cfg->logit_softcap);

  // Array fields
  printf("  %-*s: ", width, "att_layers");
  for (int i = 0; i < cfg->n_layers; i++)
  {
    printf("%d", cfg->att_layers[i] ? 1 : 0);
    if ((i + 1) % (width - 1) == 0 && i + 1 < cfg->n_layers)
    {
      printf("\n");
      for (int j = 0; j < width + 4; j++)
      {
        printf(" ");
      }
    }
  }
  printf("\n");

  // Boolean fields
  printf(
      "  %-*s: %s\n", width, "support_mm", cfg->support_mm ? "true" : "false");
  printf("  %-*s: %s\n", width, "qk_norm", cfg->qk_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "pre_mlp_norm",
      cfg->pre_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "pst_mlp_norm",
      cfg->pst_mlp_norm ? "true" : "false");
  printf("  %-*s: %s\n", width, "quant", model->quant ? "true" : "false");

  // Tokenizer
  printf("\nTokenizer:\n");
  printf("  %-*s: %d\n", width, "vocab_size", tok->vocab_size);
  printf("  %-*s: %d\n", width, "n_merges", tok->n_merges);
  printf("  %-*s: %d\n", width, "bos", tok->bos);
  printf("  %-*s: %d\n", width, "eos", tok->eos);
  printf("  %-*s: %d\n", width, "sot", tok->sot);
  printf("  %-*s: %d\n", width, "eot", tok->eot);
  printf("  %-*s: %d\n", width, "soi", tok->soi);
  printf("  %-*s: %d\n", width, "eoi", tok->eoi);
  printf("  %-*s: %d\n", width, "ist", tok->ist);

  printf("\nMemory Footprint (estimated):\n");

  size_t total = 0;

  int C   = cfg->embed_dim;
  int CH  = cfg->head_dim;
  int NH  = cfg->n_heads;
  int Cq  = NH * CH;
  int Ckv = cfg->n_kv_heads * CH;
  int CM  = cfg->mlp_dim;

  // Embedding
  if (!model->quant)
  {
    total += C * cfg->vocab_size * sizeof(floatx);
  }
  else
  {
    total += C * cfg->vocab_size * sizeof(int8_t);
    total += cfg->vocab_size * sizeof(floatx);  // scales
  }

  // Weights per layer
  for (int l = 0; l < cfg->n_layers; l++)
  {
    int layer_params = C * Cq + C * Ckv + C * Ckv + Cq * C + C * CM * 3;

    if (!model->quant)
    {
      total += layer_params * sizeof(floatx);
    }
    else
    {
      total += layer_params * sizeof(int8_t);
      total += (Cq + Ckv + Ckv + C * 2 + CM * 2) * sizeof(floatx);  // scales
    }

    // Norm layers
    total += C * sizeof(floatx);  // n1
    total += C * sizeof(floatx);  // n2
    if (cfg->qk_norm)
    {
      total += 2 * cfg->head_dim * sizeof(floatx);
    }
    if (cfg->pre_mlp_norm)
    {
      total += C * sizeof(floatx);
    }
    if (cfg->pst_mlp_norm)
    {
      total += C * sizeof(floatx);
    }
  }

  // Final norm
  total += C * sizeof(floatx);

  printf("  %-*s: %.2f GB\n", width, "Weights",
      (float)total / (1024.0 * 1024.0 * 1024.0));

  // KV Cache
  size_t kv_cache_bytes = cfg->n_layers * 2 * Ckv * sizeof(floatx);
  printf("  %-*s: %.2f KB\n", width, "KV Cache (tok)",
      (float)kv_cache_bytes / 1024.0);

  // Gemma Buffer
  int           ppi  = 0;
  VisionConfig *vcfg = NULL;
  if (cfg->support_mm && enable_mm && enc != NULL)
  {
    vcfg = enc->config;
    ppi  = vcfg->image_size / vcfg->patch_size;
  }
  int mult = (cfg->support_mm && enable_mm && enc != NULL) ? ppi * ppi : 1;

  size_t dec_bytes = 0;

  // Quantized buffers
  if (model->quant)
  {
    dec_bytes += C * sizeof(int8_t);   // x_i8
    dec_bytes += Cq * sizeof(int8_t);  // xo_i8
    dec_bytes += CM * sizeof(int8_t);  // xg_i8
  }

  // Main buffers
  dec_bytes += cfg->n_layers * 2 * seqlen * Ckv * sizeof(floatx);  // kv_cache
  dec_bytes += cfg->vocab_size * sizeof(floatx);                   // logits
  dec_bytes += mult * C * sizeof(floatx);                          // x
  dec_bytes += mult * C * sizeof(floatx);                          // resid
  dec_bytes += mult * Cq * sizeof(floatx);                         // xq
  dec_bytes += mult * Ckv * sizeof(floatx);                        // xk
  dec_bytes += mult * CH * sizeof(floatx);           // csfreqs_slid
  dec_bytes += mult * CH * sizeof(floatx);           // csfreqs_full
  dec_bytes += mult * Ckv * sizeof(floatx);          // xv
  dec_bytes += mult * Cq * sizeof(floatx);           // xo
  dec_bytes += mult * NH * seqlen * sizeof(floatx);  // att
  dec_bytes += mult * CM * sizeof(floatx);           // xg
  dec_bytes += mult * CM * sizeof(floatx);           // xu

  printf("  %-*s: %.2f MB\n", width, "Gemma Buffer",
      (float)dec_bytes / (1024.0 * 1024.0));

  // SigLIP Buffer (if applicable)
  if (cfg->support_mm && enable_mm && enc != NULL)
  {
    int C  = vcfg->hidden_dim;
    int CM = vcfg->mlp_dim;
    int N  = ppi * ppi;
    int NH = vcfg->n_heads;

    size_t enc_bytes = 0;

    // Quantized buffers
    if (model->quant)
    {
      enc_bytes += N * C * sizeof(int8_t);   // x_i8
      enc_bytes += N * sizeof(floatx);       // x_scales
      enc_bytes += N * CM * sizeof(int8_t);  // mlp_i8
      enc_bytes += N * sizeof(floatx);       // mlp_scales
    }

    // Main buffers
    enc_bytes += N * C * sizeof(floatx);       // x
    enc_bytes += N * C * sizeof(floatx);       // resid
    enc_bytes += N * C * sizeof(floatx);       // xq
    enc_bytes += N * C * sizeof(floatx);       // xk
    enc_bytes += N * C * sizeof(floatx);       // xv
    enc_bytes += N * C * sizeof(floatx);       // att_out
    enc_bytes += N * CM * sizeof(floatx);      // mlp_hidden
    enc_bytes += NH * N * N * sizeof(floatx);  // scores

    printf("  %-*s: %.2f MB\n", width, "SigLIP Buffer",
        (float)enc_bytes / (1024.0 * 1024.0));
  }

  printf("=========================================\n\n");
}

/* */
void
print_usage(void)
{
  // clang-format off
  printf(
  "usage:\n"
  "  ./gemma <modelfile> [options]\n"
  "\n"
  "arguments:\n"
  "  modelfile              path to the model file\n"
  "\n"
  "options:\n"
  "  -l, --seqlen <N>       set sequence length"
                            " (default: " TOSTRING(DEFAULT_SEQLEN) ")\n"
  "  -k, --topk <N>         set top-k sampling value"
                            " (default: " TOSTRING(DEFAULT_TOPK) ")\n"
  "  -s, --seed <N>         set random seed"
                            " (default: current time)\n"
  "  -u, --chunk <N>        set prefilling chunk size, must be >= 1"
                            " (default: " TOSTRING(DEFAULT_CHUNK_SIZE) ")\n"
  "  -t, --temperature <F>  set temperature value, must be >= 0.0"
                            " (default: " TOSTRING(DEFAULT_TEMPERATURE) ")\n"
  "  -p, --topp <F>         set top-p sampling value, must be 0.0 < p <= 1.0"
                            " (default: " TOSTRING(DEFAULT_TOPP) ")\n"
  "  -r, --rpen <F>         set repetition penalty, must be >= 1.0"
                            " (default: " TOSTRING(DEFAULT_RPEN) ")\n"
  "  -i, --prompt <S>       set input prompt, ignored if chat mode is enabled"
                            " (default: \"" DEFAULT_PROMPT "\")\n"
  "  -c, --chat             enable chat mode\n"
  "  -d, --disable-mm       disable multimodal capability\n"
  "  -v, --verbose          print model info\n"
  "  -h, --help, -?         display this help message\n"
  "\n"
  "controls:\n"
  "  Ctrl+C                 gracefully interrupt generation and exit\n"
  "\n"
  "examples:\n"
  "  ./gemma model.bin -l 2048 -t 0.8 -c\n"
  "  ./gemma model.bin -i \"Hello I'm a language model,\""
          "--seqlen 4096 --topk 50 --seed 12345\n");
  // clang-format on
}

/* */
static const char *
safe_get_arg(int i, int argc, char **argv)
{
  if (i + 1 >= argc)
  {
    print_usage();
    fprintf(stderr, "error: option '%s' requires an argument.\n", argv[i]);
    return NULL;
  }
  return argv[i + 1];
}

/* */
int
main(int argc, char **argv)
{
  set_utf8_console();
  setup_signal_handler();

  unsigned int seqlen      = DEFAULT_SEQLEN;
  unsigned int topk        = DEFAULT_TOPK;
  unsigned int seed        = (unsigned int)time(NULL);
  unsigned int chunk_size  = DEFAULT_CHUNK_SIZE;
  float        temperature = DEFAULT_TEMPERATURE;
  float        topp        = DEFAULT_TOPP;
  float        rpen        = DEFAULT_RPEN;
  const char  *prompt      = DEFAULT_PROMPT;
  bool         chatmode    = false;
  bool         enable_mm   = true;
  bool         print_cfg   = false;

  GemmaModel   *model = NULL;
  TextConfig   *cfg   = NULL;
  VisionConfig *vcfg  = NULL;
  TextBuffer   *buf   = NULL;
  VisionBuffer *vbuf  = NULL;

  // On Windows, get UTF-8 encoded command line arguments
  char **utf8_argv = get_utf8_argv(&argc);
  if (utf8_argv != NULL)
  {
    argv = utf8_argv;
  }

  if (argc < 2)
  {
    print_usage();
    fprintf(stderr, "\nerror: model filename is not provided\n");
    goto fail;
  }

  char *modelfile = argv[1];
  if (strcmp(modelfile, "-h") == 0 || strcmp(modelfile, "--help") == 0 ||
      strcmp(modelfile, "-?") == 0)
  {
    print_usage();
    goto end;
  }

  const char *val;

  // Parse the command line arguments
  for (int i = 2; i < argc; i++)
  {
    const char *arg = argv[i];

    if (strcmp(arg, "-l") == 0 || strcmp(arg, "--seqlen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seqlen))
      {
        fprintf(stderr, "error: invalid number for --seqlen: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-k") == 0 || strcmp(arg, "--topk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &topk))
      {
        fprintf(stderr, "error: invalid number for --topk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--seed") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &seed))
      {
        fprintf(stderr, "error: invalid number for --seed: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-u") == 0 || strcmp(arg, "--chunk") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atoui(val, &chunk_size) || chunk_size == 0)
      {
        fprintf(stderr, "error: invalid number for --chunk: %s\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-t") == 0 || strcmp(arg, "--temperature") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &temperature) || temperature < 0.0f)
      {
        fprintf(stderr, "error: invalid temperature: %s (must be >= 0)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--topp") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &topp) || topp <= 0.0f || topp > 1.0f)
      {
        fprintf(stderr, "error: invalid top-p: %s (must be 0 < p <= 1)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rpen") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      if (!safe_atof(val, &rpen) || rpen < 1.0f)
      {
        fprintf(stderr,
            "error: invalid repetition penalty: %s (must be >= 1)\n", val);
        goto fail;
      }
    }
    else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--prompt") == 0)
    {
      val = safe_get_arg(i++, argc, argv);
      if (!val) goto fail;
      prompt = val;
    }
    else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--chat") == 0)
    {
      chatmode = true;
    }
    else if (strcmp(arg, "-d") == 0 || strcmp(arg, "--disable-mm") == 0)
    {
      enable_mm = false;
    }
    else if (strcmp(arg, "-v") == 0 || strcmp(arg, "--verbose") == 0)
    {
      print_cfg = true;
    }
    else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0 ||
             strcmp(arg, "-?") == 0)
    {
      print_usage();
      goto end;
    }
    else
    {
      fprintf(stderr, "error: unknown option: %s\n", arg);
      print_usage();
      goto fail;
    }
  }

  srand(seed);

  // Read model
  model = read_model(modelfile, enable_mm);
  if (model == NULL) goto fail;

  // Text config
  cfg = model->decoder->config;
  // Vision config
  if (model->encoder != NULL)
  {
    vcfg = model->encoder->config;
  }

  // Text buffer
  buf = malloc_text_buffer(
      cfg, vcfg, (int)seqlen, (int)chunk_size, enable_mm, model->quant);
  if (buf == NULL) goto fail;

  // Vision buffer
  if (enable_mm && cfg->support_mm)
  {
    vbuf = malloc_vision_buffer(vcfg, model->quant);
    if (vbuf == NULL) goto fail;
  }

  if (print_cfg)
  {
    print_model_config(model, (int)seqlen, enable_mm);
  }

  if (chatmode)
  {
    chat(model, buf, vbuf, (int)seqlen, (int)chunk_size, temperature, (int)topk,
        topp, rpen, enable_mm);
  }
  else
  {
    generate(model, buf, vbuf, prompt, (int)seqlen, (int)chunk_size,
        temperature, (int)topk, topp, rpen, enable_mm);
  }

  if (g_interrupted)
  {
    printf("\ninterrupted by user\n");
  }

end:
  free_utf8_argv(utf8_argv, argc);
  if (model != NULL)
  {
    free_text_buffer(buf, model->quant);
    free_vision_buffer(vbuf, model->quant);
  }
  free_gemma_model(model);
  return 0;

fail:
  free_utf8_argv(utf8_argv, argc);
  if (model != NULL)
  {
    free_text_buffer(buf, model->quant);
    free_vision_buffer(vbuf, model->quant);
  }
  free_gemma_model(model);
  return 1;
}
