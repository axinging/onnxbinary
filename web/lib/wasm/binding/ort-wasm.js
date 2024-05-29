
var ortWasm = (() => {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
function(moduleArg = {}) {

var e=moduleArg,k,l;e.ready=new Promise((a,b)=>{k=a;l=b});var q=Object.assign({},e),v="./this.program",aa="object"==typeof window,x="function"==typeof importScripts,ba="object"==typeof process&&"object"==typeof process.versions&&"string"==typeof process.versions.node,y="",A,B,C;
if(ba){var fs=require("fs"),D=require("path");y=x?D.dirname(y)+"/":__dirname+"/";A=(a,b)=>{a=a.startsWith("file://")?new URL(a):D.normalize(a);return fs.readFileSync(a,b?void 0:"utf8")};C=a=>{a=A(a,!0);a.buffer||(a=new Uint8Array(a));return a};B=(a,b,c,f=!0)=>{a=a.startsWith("file://")?new URL(a):D.normalize(a);fs.readFile(a,f?void 0:"utf8",(g,h)=>{g?c(g):b(f?h.buffer:h)})};!e.thisProgram&&1<process.argv.length&&(v=process.argv[1].replace(/\\/g,"/"));process.argv.slice(2);e.inspect=()=>"[Emscripten Module object]"}else if(aa||
x)x?y=self.location.href:"undefined"!=typeof document&&document.currentScript&&(y=document.currentScript.src),_scriptDir&&(y=_scriptDir),0!==y.indexOf("blob:")?y=y.substr(0,y.replace(/[?#].*/,"").lastIndexOf("/")+1):y="",A=a=>{var b=new XMLHttpRequest;b.open("GET",a,!1);b.send(null);return b.responseText},x&&(C=a=>{var b=new XMLHttpRequest;b.open("GET",a,!1);b.responseType="arraybuffer";b.send(null);return new Uint8Array(b.response)}),B=(a,b,c)=>{var f=new XMLHttpRequest;f.open("GET",a,!0);f.responseType=
"arraybuffer";f.onload=()=>{200==f.status||0==f.status&&f.response?b(f.response):c()};f.onerror=c;f.send(null)};var ca=e.print||console.log.bind(console),E=e.printErr||console.error.bind(console);Object.assign(e,q);q=null;e.thisProgram&&(v=e.thisProgram);var F;e.wasmBinary&&(F=e.wasmBinary);var noExitRuntime=e.noExitRuntime||!0;"object"!=typeof WebAssembly&&G("no native wasm support detected");var H,I,da=!1,J,K,L,M;
function ea(){var a=H.buffer;e.HEAP8=J=new Int8Array(a);e.HEAP16=new Int16Array(a);e.HEAP32=L=new Int32Array(a);e.HEAPU8=K=new Uint8Array(a);e.HEAPU16=new Uint16Array(a);e.HEAPU32=M=new Uint32Array(a);e.HEAPF32=new Float32Array(a);e.HEAPF64=new Float64Array(a)}var fa=[],ha=[],ia=[];function ja(){var a=e.preRun.shift();fa.unshift(a)}var N=0,O=null,P=null;
function G(a){if(e.onAbort)e.onAbort(a);a="Aborted("+a+")";E(a);da=!0;a=new WebAssembly.RuntimeError(a+". Build with -sASSERTIONS for more info.");l(a);throw a;}function ka(a){return a.startsWith("data:application/octet-stream;base64,")}var Q;Q="ort-wasm.wasm";if(!ka(Q)){var la=Q;Q=e.locateFile?e.locateFile(la,y):y+la}function ma(a){if(a==Q&&F)return new Uint8Array(F);if(C)return C(a);throw"both async and sync fetching of the wasm failed";}
function na(a){if(!F&&(aa||x)){if("function"==typeof fetch&&!a.startsWith("file://"))return fetch(a,{credentials:"same-origin"}).then(b=>{if(!b.ok)throw"failed to load wasm binary file at '"+a+"'";return b.arrayBuffer()}).catch(()=>ma(a));if(B)return new Promise((b,c)=>{B(a,f=>b(new Uint8Array(f)),c)})}return Promise.resolve().then(()=>ma(a))}function oa(a,b,c){return na(a).then(f=>WebAssembly.instantiate(f,b)).then(f=>f).then(c,f=>{E("failed to asynchronously prepare wasm: "+f);G(f)})}
function pa(a,b){var c=Q;return F||"function"!=typeof WebAssembly.instantiateStreaming||ka(c)||c.startsWith("file://")||ba||"function"!=typeof fetch?oa(c,a,b):fetch(c,{credentials:"same-origin"}).then(f=>WebAssembly.instantiateStreaming(f,a).then(b,function(g){E("wasm streaming compile failed: "+g);E("falling back to ArrayBuffer instantiation");return oa(c,a,b)}))}var R,S=a=>{for(;0<a.length;)a.shift()(e)};
function qa(a){this.va=a-24;this.Ea=function(b){M[this.va+4>>2>>>0]=b};this.za=function(b){M[this.va+8>>2>>>0]=b};this.xa=function(b,c){this.ya();this.Ea(b);this.za(c)};this.ya=function(){M[this.va+16>>2>>>0]=0}}
var ra=0,sa=0,ta="undefined"!=typeof TextDecoder?new TextDecoder("utf8"):void 0,ua=(a,b,c)=>{b>>>=0;var f=b+c;for(c=b;a[c]&&!(c>=f);)++c;if(16<c-b&&a.buffer&&ta)return ta.decode(a.subarray(b,c));for(f="";b<c;){var g=a[b++];if(g&128){var h=a[b++]&63;if(192==(g&224))f+=String.fromCharCode((g&31)<<6|h);else{var m=a[b++]&63;g=224==(g&240)?(g&15)<<12|h<<6|m:(g&7)<<18|h<<12|m<<6|a[b++]&63;65536>g?f+=String.fromCharCode(g):(g-=65536,f+=String.fromCharCode(55296|g>>10,56320|g&1023))}}else f+=String.fromCharCode(g)}return f},
T=(a,b)=>(a>>>=0)?ua(K,a,b):"",U=a=>{for(var b=0,c=0;c<a.length;++c){var f=a.charCodeAt(c);127>=f?b++:2047>=f?b+=2:55296<=f&&57343>=f?(b+=4,++c):b+=3}return b},V=(a,b,c,f)=>{c>>>=0;if(!(0<f))return 0;var g=c;f=c+f-1;for(var h=0;h<a.length;++h){var m=a.charCodeAt(h);if(55296<=m&&57343>=m){var r=a.charCodeAt(++h);m=65536+((m&1023)<<10)|r&1023}if(127>=m){if(c>=f)break;b[c++>>>0]=m}else{if(2047>=m){if(c+1>=f)break;b[c++>>>0]=192|m>>6}else{if(65535>=m){if(c+2>=f)break;b[c++>>>0]=224|m>>12}else{if(c+3>=
f)break;b[c++>>>0]=240|m>>18;b[c++>>>0]=128|m>>12&63}b[c++>>>0]=128|m>>6&63}b[c++>>>0]=128|m&63}}b[c>>>0]=0;return c-g},W=a=>0===a%4&&(0!==a%100||0===a%400),va=[0,31,60,91,121,152,182,213,244,274,305,335],wa=[0,31,59,90,120,151,181,212,243,273,304,334],Ba=a=>{var b=U(a)+1,c=Aa(b);c&&V(a,K,c,b);return c},X={},Ca=()=>{if(!Y){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"==typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-",
"_")+".UTF-8",_:v||"./this.program"},b;for(b in X)void 0===X[b]?delete a[b]:a[b]=X[b];var c=[];for(b in a)c.push(`${b}=${a[b]}`);Y=c}return Y},Y,Da=[null,[],[]],Ea=[31,29,31,30,31,30,31,31,30,31,30,31],Fa=[31,28,31,30,31,30,31,31,30,31,30,31];function Ga(a){var b=Array(U(a)+1);V(a,b,0,b.length);return b}
function Ha(a,b,c,f){function g(d,n,p){for(d="number"==typeof d?d.toString():d||"";d.length<n;)d=p[0]+d;return d}function h(d,n){return g(d,n,"0")}function m(d,n){function p(xa){return 0>xa?-1:0<xa?1:0}var z;0===(z=p(d.getFullYear()-n.getFullYear()))&&0===(z=p(d.getMonth()-n.getMonth()))&&(z=p(d.getDate()-n.getDate()));return z}function r(d){switch(d.getDay()){case 0:return new Date(d.getFullYear()-1,11,29);case 1:return d;case 2:return new Date(d.getFullYear(),0,3);case 3:return new Date(d.getFullYear(),
0,2);case 4:return new Date(d.getFullYear(),0,1);case 5:return new Date(d.getFullYear()-1,11,31);case 6:return new Date(d.getFullYear()-1,11,30)}}function w(d){var n=d.ra;for(d=new Date((new Date(d.sa+1900,0,1)).getTime());0<n;){var p=d.getMonth(),z=(W(d.getFullYear())?Ea:Fa)[p];if(n>z-d.getDate())n-=z-d.getDate()+1,d.setDate(1),11>p?d.setMonth(p+1):(d.setMonth(0),d.setFullYear(d.getFullYear()+1));else{d.setDate(d.getDate()+n);break}}p=new Date(d.getFullYear()+1,0,4);n=r(new Date(d.getFullYear(),
0,4));p=r(p);return 0>=m(n,d)?0>=m(p,d)?d.getFullYear()+1:d.getFullYear():d.getFullYear()-1}a>>>=0;b>>>=0;c>>>=0;f>>>=0;var t=L[f+40>>2>>>0];f={Ca:L[f>>2>>>0],Ba:L[f+4>>2>>>0],ta:L[f+8>>2>>>0],wa:L[f+12>>2>>>0],ua:L[f+16>>2>>>0],sa:L[f+20>>2>>>0],ma:L[f+24>>2>>>0],ra:L[f+28>>2>>>0],Fa:L[f+32>>2>>>0],Aa:L[f+36>>2>>>0],Da:t?T(t):""};c=T(c);t={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c",
"%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var u in t)c=c.replace(new RegExp(u,"g"),t[u]);var ya="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),za="January February March April May June July August September October November December".split(" ");t={"%a":d=>ya[d.ma].substring(0,3),"%A":d=>ya[d.ma],"%b":d=>
za[d.ua].substring(0,3),"%B":d=>za[d.ua],"%C":d=>h((d.sa+1900)/100|0,2),"%d":d=>h(d.wa,2),"%e":d=>g(d.wa,2," "),"%g":d=>w(d).toString().substring(2),"%G":d=>w(d),"%H":d=>h(d.ta,2),"%I":d=>{d=d.ta;0==d?d=12:12<d&&(d-=12);return h(d,2)},"%j":d=>{for(var n=0,p=0;p<=d.ua-1;n+=(W(d.sa+1900)?Ea:Fa)[p++]);return h(d.wa+n,3)},"%m":d=>h(d.ua+1,2),"%M":d=>h(d.Ba,2),"%n":()=>"\n","%p":d=>0<=d.ta&&12>d.ta?"AM":"PM","%S":d=>h(d.Ca,2),"%t":()=>"\t","%u":d=>d.ma||7,"%U":d=>h(Math.floor((d.ra+7-d.ma)/7),2),"%V":d=>
{var n=Math.floor((d.ra+7-(d.ma+6)%7)/7);2>=(d.ma+371-d.ra-2)%7&&n++;if(n)53==n&&(p=(d.ma+371-d.ra)%7,4==p||3==p&&W(d.sa)||(n=1));else{n=52;var p=(d.ma+7-d.ra-1)%7;(4==p||5==p&&W(d.sa%400-1))&&n++}return h(n,2)},"%w":d=>d.ma,"%W":d=>h(Math.floor((d.ra+7-(d.ma+6)%7)/7),2),"%y":d=>(d.sa+1900).toString().substring(2),"%Y":d=>d.sa+1900,"%z":d=>{d=d.Aa;var n=0<=d;d=Math.abs(d)/60;return(n?"+":"-")+String("0000"+(d/60*100+d%60)).slice(-4)},"%Z":d=>d.Da,"%%":()=>"%"};c=c.replace(/%%/g,"\x00\x00");for(u in t)c.includes(u)&&
(c=c.replace(new RegExp(u,"g"),t[u](f)));c=c.replace(/\0\0/g,"%");u=Ga(c);if(u.length>b)return 0;J.set(u,a>>>0);return u.length-1}
var Ja={a:function(a,b,c){a>>>=0;(new qa(a)).xa(b>>>0,c>>>0);ra=a;sa++;throw ra;},e:function(){return 0},H:function(){},x:function(){},z:function(){},k:function(){return 0},F:function(){},B:function(){},E:function(){},g:function(){},y:function(){},v:function(){},G:function(){},w:function(){},l:()=>!0,o:function(a,b,c){a=b+2097152>>>0<4194305-!!a?(a>>>0)+4294967296*b:NaN;c>>>=0;a=new Date(1E3*a);L[c>>2>>>0]=a.getUTCSeconds();L[c+4>>2>>>0]=a.getUTCMinutes();L[c+8>>2>>>0]=a.getUTCHours();L[c+12>>2>>>
0]=a.getUTCDate();L[c+16>>2>>>0]=a.getUTCMonth();L[c+20>>2>>>0]=a.getUTCFullYear()-1900;L[c+24>>2>>>0]=a.getUTCDay();L[c+28>>2>>>0]=(a.getTime()-Date.UTC(a.getUTCFullYear(),0,1,0,0,0,0))/864E5|0},p:function(a,b,c){a=b+2097152>>>0<4194305-!!a?(a>>>0)+4294967296*b:NaN;c>>>=0;a=new Date(1E3*a);L[c>>2>>>0]=a.getSeconds();L[c+4>>2>>>0]=a.getMinutes();L[c+8>>2>>>0]=a.getHours();L[c+12>>2>>>0]=a.getDate();L[c+16>>2>>>0]=a.getMonth();L[c+20>>2>>>0]=a.getFullYear()-1900;L[c+24>>2>>>0]=a.getDay();L[c+28>>2>>>
0]=(W(a.getFullYear())?va:wa)[a.getMonth()]+a.getDate()-1|0;L[c+36>>2>>>0]=-(60*a.getTimezoneOffset());b=(new Date(a.getFullYear(),6,1)).getTimezoneOffset();var f=(new Date(a.getFullYear(),0,1)).getTimezoneOffset();L[c+32>>2>>>0]=(b!=f&&a.getTimezoneOffset()==Math.min(f,b))|0},q:function(a){a>>>=0;var b=new Date(L[a+20>>2>>>0]+1900,L[a+16>>2>>>0],L[a+12>>2>>>0],L[a+8>>2>>>0],L[a+4>>2>>>0],L[a>>2>>>0],0),c=L[a+32>>2>>>0],f=b.getTimezoneOffset(),g=(new Date(b.getFullYear(),6,1)).getTimezoneOffset(),
h=(new Date(b.getFullYear(),0,1)).getTimezoneOffset(),m=Math.min(h,g);0>c?L[a+32>>2>>>0]=Number(g!=h&&m==f):0<c!=(m==f)&&(g=Math.max(h,g),b.setTime(b.getTime()+6E4*((0<c?m:g)-f)));L[a+24>>2>>>0]=b.getDay();L[a+28>>2>>>0]=(W(b.getFullYear())?va:wa)[b.getMonth()]+b.getDate()-1|0;L[a>>2>>>0]=b.getSeconds();L[a+4>>2>>>0]=b.getMinutes();L[a+8>>2>>>0]=b.getHours();L[a+12>>2>>>0]=b.getDate();L[a+16>>2>>>0]=b.getMonth();L[a+20>>2>>>0]=b.getYear();a=b.getTime()/1E3;return Ia((R=a,1<=+Math.abs(R)?0<R?+Math.floor(R/
4294967296)>>>0:~~+Math.ceil((R-+(~~R>>>0))/4294967296)>>>0:0)),a>>>0},m:function(){return-52},n:function(){},t:function(a,b,c){function f(w){return(w=w.toTimeString().match(/\(([A-Za-z ]+)\)$/))?w[1]:"GMT"}c>>>=0;var g=(new Date).getFullYear(),h=new Date(g,0,1),m=new Date(g,6,1);g=h.getTimezoneOffset();var r=m.getTimezoneOffset();M[a>>>0>>2>>>0]=60*Math.max(g,r);L[b>>>0>>2>>>0]=Number(g!=r);a=f(h);b=f(m);a=Ba(a);b=Ba(b);r<g?(M[c>>2>>>0]=a,M[c+4>>2>>>0]=b):(M[c>>2>>>0]=b,M[c+4>>2>>>0]=a)},d:()=>{G("")},
h:function(){return Date.now()},u:function(){return 4294901760},b:()=>performance.now(),I:function(a,b,c){b>>>=0;return K.copyWithin(a>>>0>>>0,b>>>0,b+(c>>>0)>>>0)},s:function(a){a>>>=0;var b=K.length;if(4294901760<a)return!1;for(var c=1;4>=c;c*=2){var f=b*(1+.2/c);f=Math.min(f,a+100663296);var g=Math;f=Math.max(a,f);a:{g=g.min.call(g,4294901760,f+(65536-f%65536)%65536)-H.buffer.byteLength+65535>>>16;try{H.grow(g);ea();var h=1;break a}catch(m){}h=void 0}if(h)return!0}return!1},C:function(a,b){a>>>=
0;b>>>=0;var c=0;Ca().forEach(function(f,g){var h=b+c;g=M[a+4*g>>2>>>0]=h;for(h=0;h<f.length;++h)J[g++>>0>>>0]=f.charCodeAt(h);J[g>>0>>>0]=0;c+=f.length+1});return 0},D:function(a,b){a>>>=0;b>>>=0;var c=Ca();M[a>>2>>>0]=c.length;var f=0;c.forEach(function(g){f+=g.length+1});M[b>>2>>>0]=f;return 0},f:()=>52,j:function(){return 52},r:function(){return 70},i:function(a,b,c,f){b>>>=0;c>>>=0;f>>>=0;for(var g=0,h=0;h<c;h++){var m=M[b>>2>>>0],r=M[b+4>>2>>>0];b+=8;for(var w=0;w<r;w++){var t=K[m+w>>>0],u=
Da[a];0===t||10===t?((1===a?ca:E)(ua(u,0)),u.length=0):u.push(t)}g+=r}M[f>>2>>>0]=g;return 0},A:Ha,c:function(a,b,c,f){return Ha(a>>>0,b>>>0,c>>>0,f>>>0)}};
(function(){function a(c){c=c.exports;I=c=Ka(c);H=I.J;ea();ha.unshift(I.K);N--;e.monitorRunDependencies&&e.monitorRunDependencies(N);if(0==N&&(null!==O&&(clearInterval(O),O=null),P)){var f=P;P=null;f()}return c}var b={a:Ja};N++;e.monitorRunDependencies&&e.monitorRunDependencies(N);if(e.instantiateWasm)try{return e.instantiateWasm(b,a)}catch(c){E("Module.instantiateWasm callback failed with error: "+c),l(c)}pa(b,function(c){a(c.instance)}).catch(l);return{}})();
e._OrtInit=(a,b)=>(e._OrtInit=I.L)(a,b);e._OrtGetLastError=(a,b)=>(e._OrtGetLastError=I.M)(a,b);e._OrtCreateSessionOptions=(a,b,c,f,g,h,m,r,w,t)=>(e._OrtCreateSessionOptions=I.N)(a,b,c,f,g,h,m,r,w,t);e._OrtAppendExecutionProvider=(a,b)=>(e._OrtAppendExecutionProvider=I.O)(a,b);e._OrtAddFreeDimensionOverride=(a,b,c)=>(e._OrtAddFreeDimensionOverride=I.P)(a,b,c);e._OrtAddSessionConfigEntry=(a,b,c)=>(e._OrtAddSessionConfigEntry=I.Q)(a,b,c);e._OrtReleaseSessionOptions=a=>(e._OrtReleaseSessionOptions=I.R)(a);
e._OrtCreateSession=(a,b,c)=>(e._OrtCreateSession=I.S)(a,b,c);e._OrtReleaseSession=a=>(e._OrtReleaseSession=I.T)(a);e._OrtGetInputOutputCount=(a,b,c)=>(e._OrtGetInputOutputCount=I.U)(a,b,c);e._OrtGetInputName=(a,b)=>(e._OrtGetInputName=I.V)(a,b);e._OrtGetOutputName=(a,b)=>(e._OrtGetOutputName=I.W)(a,b);e._OrtFree=a=>(e._OrtFree=I.X)(a);e._OrtCreateTensor=(a,b,c,f,g,h)=>(e._OrtCreateTensor=I.Y)(a,b,c,f,g,h);e._OrtGetTensorData=(a,b,c,f,g)=>(e._OrtGetTensorData=I.Z)(a,b,c,f,g);
e._OrtReleaseTensor=a=>(e._OrtReleaseTensor=I._)(a);e._OrtCreateRunOptions=(a,b,c,f)=>(e._OrtCreateRunOptions=I.$)(a,b,c,f);e._OrtAddRunConfigEntry=(a,b,c)=>(e._OrtAddRunConfigEntry=I.aa)(a,b,c);e._OrtReleaseRunOptions=a=>(e._OrtReleaseRunOptions=I.ba)(a);e._OrtCreateBinding=a=>(e._OrtCreateBinding=I.ca)(a);e._OrtBindInput=(a,b,c)=>(e._OrtBindInput=I.da)(a,b,c);e._OrtBindOutput=(a,b,c,f)=>(e._OrtBindOutput=I.ea)(a,b,c,f);e._OrtClearBoundOutputs=a=>(e._OrtClearBoundOutputs=I.fa)(a);
e._OrtReleaseBinding=a=>(e._OrtReleaseBinding=I.ga)(a);e._OrtRunWithBinding=(a,b,c,f,g)=>(e._OrtRunWithBinding=I.ha)(a,b,c,f,g);e._OrtRun=(a,b,c,f,g,h,m,r)=>(e._OrtRun=I.ia)(a,b,c,f,g,h,m,r);e._OrtEndProfiling=a=>(e._OrtEndProfiling=I.ja)(a);var Aa=e._malloc=a=>(Aa=e._malloc=I.ka)(a);e._free=a=>(e._free=I.la)(a);var Ia=a=>(Ia=I.na)(a),La=()=>(La=I.oa)(),Ma=a=>(Ma=I.pa)(a),Na=a=>(Na=I.qa)(a);
function Ka(a){a=Object.assign({},a);var b=f=>()=>f()>>>0,c=f=>g=>f(g)>>>0;a.__errno_location=b(a.__errno_location);a.malloc=c(a.malloc);a.stackSave=b(a.stackSave);a.stackAlloc=c(a.stackAlloc);return a}e.stackAlloc=Na;e.stackSave=La;e.stackRestore=Ma;e.UTF8ToString=T;e.stringToUTF8=(a,b,c)=>V(a,K,b,c);e.lengthBytesUTF8=U;var Z;P=function Oa(){Z||Pa();Z||(P=Oa)};
function Pa(){function a(){if(!Z&&(Z=!0,e.calledRun=!0,!da)){S(ha);k(e);if(e.onRuntimeInitialized)e.onRuntimeInitialized();if(e.postRun)for("function"==typeof e.postRun&&(e.postRun=[e.postRun]);e.postRun.length;){var b=e.postRun.shift();ia.unshift(b)}S(ia)}}if(!(0<N)){if(e.preRun)for("function"==typeof e.preRun&&(e.preRun=[e.preRun]);e.preRun.length;)ja();S(fa);0<N||(e.setStatus?(e.setStatus("Running..."),setTimeout(function(){setTimeout(function(){e.setStatus("")},1);a()},1)):a())}}
if(e.preInit)for("function"==typeof e.preInit&&(e.preInit=[e.preInit]);0<e.preInit.length;)e.preInit.pop()();Pa();


  return moduleArg.ready
}

);
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = ortWasm;
else if (typeof define === 'function' && define['amd'])
  define([], () => ortWasm);
