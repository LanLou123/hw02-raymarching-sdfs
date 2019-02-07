# CIS 566 Homework 2: Implicit Surfaces

![](mata.gif)

## Features implemented:
- some SDF shapes and operations like twist or union
- Ray march optimization by checking of bounding volume
- Animation of metaballs flying around the center geometry which simply achieved by several sin and cos functions
  and also animations of the morphing of the geometry in the center by applying a smoothstep of a sin function of several different SDF and mix them together
  the gif below showed a twisted geometry
  ![](mata1.gif)
- texture of the floor :
  - the checker board feature is achieved by first computing "floor" of the UV coordinates on the plane SDF shape, then mod ```uvtile.x+uvtile.y``` with 2.f.
  - the stripe appears on top of checker board is done by applying sin function to a ramped FBM perlin noise
- specular and fresnel of the shapes using their surface normal
- subsurface scattering with self soft shadow basically achieved by shooting ray inside the geometry onece it hits it, and let it accumulate color until it exits.
- reflection of shape: the method I used to create this effect is rather simple, when a ray hit the SDF plane, I will use the hit point as a new ray's origin and shoot the ray in the 
reflected direction of the original incomming ray, and depending on the texture of the surface, some of the ray are not reflected , this is why you can see 
that the reflected scene is not contigious instead are cut by the "stripes".
- some shadow casted to the floor by the shapes.

## References:
- [Subsurface scatterig by TekF](https://www.shadertoy.com/view/4dsGRl)
- SDF functions from IQ's website
