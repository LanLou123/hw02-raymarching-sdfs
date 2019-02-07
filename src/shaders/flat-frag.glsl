#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up, u_Light;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform float u_Coherence;

in vec2 fs_Pos;
out vec4 out_Col;

#define FOV 45.f

const float epsilon = .003;
const float normalPrecision = .1;
const float shadowOffset = .1;
const int traceDepth = 500; // takes time
const float drawDistance = 100.0;

const vec3 CamPos = vec3(0,40.0,-40.0);
const vec3 CamLook = vec3(0,0,0);

const vec3 lightDir = vec3(.7,1,-.1);
const vec3 fillLightDir = vec3(0,0,-1);
const vec3 lightColour = vec3(1.1,1.05,1);
const vec3 fillLightColour = vec3(.38,.4,.42);


#define NUM_NOISE_OCTAVES 5




float shash( vec2 p )
{
	float h = dot(p,vec2(127.1,311.7));

#ifdef MOD_KazimirO
    return -1.0 + 2.0*fract(sin(h)*0.0437585453123*iTime);
#else
    return -1.0 + 2.0*fract(sin(h)*43758.5453123);
#endif
}

float snoise( in vec2 p )
{
    vec2 i = floor( p );
    vec2 f = fract( p );

	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( shash( i + vec2(0.0,0.0) ),
                     shash( i + vec2(1.0,0.0) ), u.x),
                mix( shash( i + vec2(0.0,1.0) ),
                     shash( i + vec2(1.0,1.0) ), u.x), u.y);
}

float tvv( in vec2 coord )
{
    float
     value = snoise(coord / 64.) * 64.;
    value += snoise(coord / 32.) * 32.;
    value += snoise(coord / 16.) * 16.;
    value += snoise(coord / 8.) * 8.;
    value += snoise(coord / 4.) * 4.;
    value += snoise(coord / 2.) * 2.;
    value += snoise(coord);
    value += snoise(coord / .5) * .5;
    value += snoise(coord / .25) * .25;
    return value;
}


//noise

float hash(float n) { return fract(sin(n) * 1e4); }
float hash(vec2 p) { return fract(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x)))); }
float noise(vec2 x) {
    vec2 i = floor(x);
    vec2 f = fract(x);

	// Four corners in 2D of a tile
	float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    // Simple 2D lerp using smoothstep envelope between the values.
	// return vec3(mix(mix(a, b, smoothstep(0.0, 1.0, f.x)),
	//			mix(c, d, smoothstep(0.0, 1.0, f.x)),
	//			smoothstep(0.0, 1.0, f.y)));

	// Same code, with the clamps in smoothstep and common subexpressions
	// optimized away.
    vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}
float noise(float x) {
    float i = floor(x);
    float f = fract(x);
    float u = f * f * (3.0 - 2.0 * f);
    return mix(hash(i), hash(i + 1.0), u);
}

float noise(vec3 x) {
    const vec3 step = vec3(110, 241, 171);

    vec3 i = floor(x);
    vec3 f = fract(x);

    // For performance, compute the base input to a 1D hash from the integer part of the argument and the
    // incremental change to the 1D based on the 3D -> 1D wrapping
    float n = dot(i, step);

    vec3 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(n + dot(step, vec3(0, 0, 0))), hash(n + dot(step, vec3(1, 0, 0))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 0))), hash(n + dot(step, vec3(1, 1, 0))), u.x), u.y),
               mix(mix( hash(n + dot(step, vec3(0, 0, 1))), hash(n + dot(step, vec3(1, 0, 1))), u.x),
                   mix( hash(n + dot(step, vec3(0, 1, 1))), hash(n + dot(step, vec3(1, 1, 1))), u.x), u.y), u.z);
}

float fbm(vec2 x) {
	float v = 0.0;
	float a = 0.5;
	vec2 shift = vec2(100);
	// Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
	for (int i = 0; i < NUM_NOISE_OCTAVES; ++i) {
		v += a * noise(x);
		x = rot * x * 2.0 + shift;
		a *= 0.5;
	}
	return v;
}
//

	float sdCappedCylinder( vec3 p, vec2 h )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float sphere(vec3 pos)
{
	return length(pos)-1.0;
}

float sdCylinder( vec3 p, vec3 c )
{
  return length(p.xz-c.xy)-c.z;
}

float opOnion( in float sdf, in float thickness )
{
    return abs(sdf)-thickness;
}
float opUnion( float d1, float d2 ) { return min(d1,d2); }
float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }
float opIntersection( float d1, float d2 ) { return max(d1,d2); }

float ucy(vec3 p){
vec3 c = vec3(1);
return opUnion(sdCylinder(p,c),sphere(p));
}

float box(vec3 pos)
{
    vec3 d = abs(pos) - 1.0;
  	return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

	float dot2( in vec3 v ) { return dot(v,v); }
float udQuad( vec3 p, vec3 a, vec3 b, vec3 c, vec3 d )
{
    vec3 ba = b - a; vec3 pa = p - a;
    vec3 cb = c - b; vec3 pb = p - b;
    vec3 dc = d - c; vec3 pc = p - c;
    vec3 ad = a - d; vec3 pd = p - d;
    vec3 nor = cross( ba, ad );

    return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(dc,nor),pc)) +
     sign(dot(cross(ad,nor),pd))<3.0)
     ?
     min( min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(dc*clamp(dot(dc,pc)/dot2(dc),0.0,1.0)-pc) ),
     dot2(ad*clamp(dot(ad,pd)/dot2(ad),0.0,1.0)-pd) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}
float dissphere( vec3 p)
{
    float s = length(p)-1.0;
    float dis = sin(4.f*p.x)*sin(4.f*p.y)*sin(4.f*p.z);
    return (s+dis);
}

float opTwist( in vec3 p )
{
    float c = cos(5.0*p.y);
    float s = sin(5.0*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return box(q);
}

float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

float torus(vec3 pos)
{

	vec2 q = vec2(length(pos.xz)-2.0,pos.y);
  	return length(q)-0.5;
}

float mata(float d1, float d2, float d3, float d4, float d5, float d6, float d7)
{
    float k = u_Coherence;
	return -log(exp(-k*d1)+exp(-k*d2)+exp(-k*d3)+exp(-k*d4)+exp(-k*d5)+exp(-k*d6)+exp(-k*d7))/k;
}

float scene(vec3 pos)
{
    float t = u_Time/100.f;


    //float p = torus(pos + vec3(0.0,3.0,0.0));
	float b1 =  opTwist(0.5*(pos + vec3(0.0,0.0,0.0)));
	float cy = sdCappedCylinder(1.f*pos,vec2(0,1));
	float tt = torus(pos*.5f);
	float b2 = opOnion(opOnion(opOnion(cy,0.1),0.05),0.03);
	float b =mix(b1,tt,smoothstep(0.0,1.0,cos(u_Time/200.f)));
    float s1 = sphere(2.0*(pos + 3.0 * vec3(cos(t + 1.1),sin(t + 1.1),cos(t+1.7))))/2.0;
    float s2 = sphere(2.0*(pos + 3.0 * vec3(cos(t+0.7),sin(t+0.7),cos(t+2.3))))/2.0;
    float s3 = sphere(2.0*(pos + 3.0 * vec3(cos(t+0.3),cos(t+2.9),sin(t+1.1))))/2.0;
    float s4 = sphere(2.0*(pos + 3.0 * vec3(sin(t+1.3),sin(t+1.7),sin(t+0.7))))/2.0;
    float s5 = sphere(2.0*(pos + 3.0 * vec3(sin(t+2.3),sin(t+1.9),sin(t+2.9))))/2.0;
    float p = sphere(2.0*(pos + 3.0 * vec3(sin(t+.3),sin(t+3.9),sin(t+1.9))))/2.0;
    return mata(p, b, s1, s2, s3, s4, s5);
}

float calcIntersection( in vec3 ro, in vec3 rd )
{
	const float maxd = 115.0;
	const float precis = 0.1;
    float h = precis*2.0;
    float t = 0.0;
	float res = -1.0;
    for( int i=0; i<80; i++ )
    {
        if( h<precis||t>maxd ) break;
        vec4 n = vec4(0.0,1.0,0.0,0.0);
        vec3 a = vec3(-40,-5,40);
        vec3 b = vec3(40,-5,40);
        vec3 c = vec3(40,-5,-40);
        vec3 d = vec3(-40,-5,-40);
	    h = udQuad( ro+rd*t ,a,b,c,d );
        t += h;
    }

    if( t<maxd ) res = t;
    return res;
}

vec3 calcNormal( in vec3 pos )
{
    const float eps = 0.002;

    const vec3 v1 = vec3( 1.0,-1.0,-1.0);
    const vec3 v2 = vec3(-1.0,-1.0, 1.0);
    const vec3 v3 = vec3(-1.0, 1.0,-1.0);
    const vec3 v4 = vec3( 1.0, 1.0, 1.0);

	return normalize( v1*scene( pos + v1*eps ) +
					  v2*scene( pos + v2*eps ) +
					  v3*scene( pos + v3*eps ) +
					  v4*scene( pos + v4*eps ) );
}


vec4 Shading( vec3 pos, vec3 norm, vec3 ssCol, vec3 rd )
{
	vec3 albedo = vec3(1);//mix( vec3(1,.8,.7), vec3(.5,.2,.1), Noise(pos*vec3(1,10,1)) );

    ssCol/=10.0;
	vec3 l = lightColour*mix(ssCol,vec3(1)*max(0.0,dot(norm,normalize(lightDir))),.0);
	vec3 fl = fillLightColour*(dot(norm,normalize(fillLightDir))*.5+.5);

	vec3 view = normalize(-rd);
	vec3 h = normalize(view+lightDir);
	float specular = pow(max(0.0,dot(h,norm)),2000.0);

	float fresnel = pow( 1.0 - dot( view, norm ), 5.0 );
	fresnel = mix( .01, 1.0, min(1.0,fresnel) );
	return vec4( albedo*(l+fl)*(1.0-fresnel) + ssCol*specular*2.0*lightColour, fresnel );
}

float Trace( vec3 ro, vec3 rd )
{
	float t = 0.0;
	float dist = 1.0;
	for ( int i=0; i < traceDepth; i++ )
	{
		if ( abs(dist) < epsilon || t > drawDistance || t < 0.0 )
			continue;
		dist = scene( ro+rd*t );
		t = t+dist;
	}

	// reduce edge sparkles, caused by reflections on failed positions
	if ( dist > epsilon )
		return drawDistance+1.0;

	return t;//vec4(ro+rd*t,dist);
}

vec3 SubsurfaceTrace( vec3 ro, vec3 rd )
{
	vec3 density = pow(vec3(.6,.2,.2),vec3(.4));
	const float confidence = .01;
	vec3 ssCol = vec3(1.0);

    vec3 sro = ro;

	float lastVal = scene(ro);
	float soft = 0.0;
	for ( int i=1; i < 50; i++ )
	{
		if ( ssCol.x < confidence )
			continue;

		float val = scene(ro);
		vec3 softened = pow(density,.8f*vec3(smoothstep(soft,-soft,val)));

		if ( (val-soft)*lastVal < 0.0 )
		{

			float transition = -min(val-soft,lastVal)/abs(val-soft-lastVal);
			ssCol *= pow(softened,vec3(transition));
		}
		else if ( val-soft < 0.0 )
		{
			ssCol *= softened;
		}

		soft += .05;
		lastVal = val+soft;
		ro += rd*.1;
	}



	return ssCol;
}

vec3 SkyColour( vec3 rd )
{
	// hide cracks in cube map
	//  rd -= sign(abs(rd.xyz)-abs(rd.yzx))*.01;

    rd = normalize(rd);
    vec3 FogColour = vec3(0.1,0.2,0.3);
    vec3 suncol = vec3(1);
    vec3 avs = mix( vec3(.6,.6,.6), FogColour, abs(rd.y) );
    vec3 final;
    if(dot(normalize(u_Light),rd)>.9){
    final = mix(avs,suncol,(dot(normalize(u_Light),rd) - .9)*10.f);}
    else
    final = avs;


	return final;

}




void main() {

  float sx = (2.f*gl_FragCoord.x/u_Dimensions.x)-1.f;
  float sy = 1.f-(2.f*gl_FragCoord.y/u_Dimensions.y);
  float len = length(u_Ref - u_Eye);
  vec3 forward = normalize(u_Ref - u_Eye);
  vec3 right = cross(forward,u_Up);
  vec3 V = u_Up * len * tan(FOV/2.f);
  vec3 H = right * len * (u_Dimensions.x/u_Dimensions.y) * tan(FOV/2.f);
  vec3 p = u_Ref + sx * H - sy * V;

  vec3 raydir = normalize(p - u_Eye);
  vec3 rds = raydir;

  vec3 col = vec3(0.0,0.0,0.0);

  float t = 0.f;

  vec3 lightdir = vec3 (1.f,1.f,1.f);

  vec3 ro = u_Eye;



float epsilon = 7e-2;

vec4 result = vec4(0.0,0.0,0.0,1.0);
 float t1 = Trace(ro,raydir);

vec3 efvcol = vec3(0);
   float inter = calcIntersection(ro,raydir);
   vec3 pos = ro + raydir * inter;
   vec3 checkercol = vec3(0);
   vec3 checkerref = vec3(0);
   if(inter>0.f&&t1 == (drawDistance+1.0)){

   vec2 tile = floor(pos.xz/3.0);
   checkercol =vec3(.4,.4,.4)*mod((tile.x+tile.y),2.0) ;


   float efv = cos(tvv(vec2(pos.x*4.0,pos.z*4.0)));

    efvcol= vec3(1)*efv;


   checkercol += 0.1*clamp(efvcol,0.0,1.0);

   ro = pos;
   vec3 n = vec3(0,1,0);
   if(efv>0.f)
   raydir = reflect(raydir,n);
   float nx = noise(raydir.x*10.f);
   float ny = noise(raydir.y*3.f);
   float nz = noise(raydir.z*5.f);
   raydir = raydir - 0.01*vec3(nx,ny,nz);

   t1 = Trace(ro,raydir);

   }




 ro = ro+raydir*t1;
 vec3 norm = calcNormal(ro);



 vec3 viewdir = raydir;

    float fbias = -0.1;
    float fpow = 15.0;
    float fscale = 1.0;

 float R = max(0.0, min(1.0, fbias + fscale * pow(1.0 + dot(rds, norm), fpow)));//emperical fresnel

 vec3 fcol = vec3(R);

 vec3 halfway = normalize(u_Light - viewdir);
 float spec = pow(max(0.0,dot(halfway,normalize(norm))),160.f);
 ro = raydir*epsilon+ro;
 vec3 ss = SubsurfaceTrace(ro,u_Light);
 vec3 speccol = vec3(clamp(spec * ss.x - 0.4,0.0,1.0));
 result = vec4(ss + speccol + checkercol,1.0);

 if(inter<0.f&&t1==(drawDistance + 1.0)){
 result = vec4(SkyColour(rds),1.0);
 }
 if(inter>0.f&&t1==(drawDistance + 1.0)){


 float shadowv  = Trace(pos,u_Light);
 vec3 shadow = vec3(0);
 if(shadowv!=(drawDistance+1.0))
 {shadow = vec3(-0.4,-0.4,-0.4);}

 if(efvcol.x>0.f)
 checkerref = SkyColour(raydir);


 result = vec4(checkercol+checkerref+shadow,1.0);
 }

  out_Col = result;//vec4(col,1.f);//vec4(0.5 * (fs_Pos + vec2(1.0)), 0.5 * (sin(u_Time * 3.14159 * 0.01) + 1.0), 1.0);
}
