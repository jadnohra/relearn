from ctypes import *
import math
import time
import copy
import random
import pickle
# conda install pyopengl
# conda install -c conda-forge freeglut
# or: brew install freeglut
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
#
g_dbg = '-dbg' in sys.argv
#
def sys_argv_has(keys):
	if (hasattr(sys, 'argv')):
	 return any(x in sys.argv for x in keys)
	return False
def sys_argv_has_key(keys):
	if ( hasattr(sys, 'argv')):
		for key in keys:
			ki = sys.argv.index(key) if key in sys.argv else -1
			if (ki >= 0 and ki+1 < len(sys.argv)):
				return True
	return False
def sys_argv_get(keys, dflt):
	if ( hasattr(sys, 'argv')):
		for key in keys:
			ki = sys.argv.index(key) if key in sys.argv else -1
			if (ki >= 0 and ki+1 < len(sys.argv)):
				return sys.argv[ki+1]
	return dflt
#
def m_min(v1, v2):
	return v1 if (v1 <= v2) else v2
def m_max(v1, v2):
	return v1 if (v1 >= v2) else v2
def m_interp(v1, v2, t):
	return v1*(1.0-t)+v2*(t)
def m_abs(v):
	return v if (v >= 0) else -v
def v2_p(x,y):
	return [x,y]
def v2_eq(x,y):
	return x[0] == y[0] and x[1] == y[1]
def v2_z():
	return [0.0,0.0]
def v2_sz(v):
	v[0] = 0.0
	v[1] = 0.0
def v2_add(v1, v2):
	return [v1[0]+v2[0], v1[1]+v2[1]]
def v2_sub(v1, v2):
	return [v1[0]-v2[0], v1[1]-v2[1]]
def v2_dot(v1, v2):
	return v1[0]*v2[0]+v1[1]*v2[1]
def v2_lenSq(v1):
	return v2_dot(v1, v1)
def v2_len(v1):
	return math.sqrt(v2_lenSq(v1))
def v2_distSq(p1, p2):
	vec = v2_sub(p2, p1)
	return v2_lenSq(vec)
def v2_dist(p1, p2):
	vec = v2_sub(p2, p1)
	return v2_len(vec)
def v2_muls(v1, s):
	return [v1[0]*s, v1[1]*s]
def v2_neg(v1):
	return [ -v1[0], -v1[1] ]
def v2_copy(v):
	return [v[0], v[1]]
def v2_interp(v1, v2, t):
	return [m_interp(v1[i], v2[i], t[i]) for i in range(2)]
def v2_normalize(v1):
	l = v2_len(v1)
	if l != 0.0:
		return v2_muls(v1, 1.0/l)
	return v1
def v2_v_angle(vec):
	return math.atan2(vec[1], vec[0])
def v2_p_angle(v1,v2):
	return v2_v_angle(v2_sub(v2, v1))
def v2_rot(v1,a):
	c = math.cos(a)
	s = math.sin(a)
	ret = [ v1[0]*c + v1[1]*(-s), v1[0]*s + v1[1]*(c) ]
	return ret
def v2_orth(v1):
	return [ -v1[1], v1[0] ]
def v2_rot90(v1):
	return [ -v1[1], v1[0] ]
def v2_rotm90(v1):
	return [ v1[1], -v1[0] ]
def v2_projs(v, a):
	return v2_dot(v, a) / v2_dot(a, a)
def v2_proj(v, a):
	return v2_muls(a, v2_projs(v, a))
def v2_points_proj(a, b, c):
	return v2_add(a, v2_proj(v2_sub(c, a), v2_sub(b, a)))
def v2_proj_rest(v, a):
	return v2_sub(v, v2_proj(v, a))
def v2_points_proj_rest(a, b, c):
	return v2_proj_rest(v2_sub(c, a), v2_sub(b, a))
def Matrix(r,c,v):
	return [[v]*c for x in xrange(r)]
def Eye(r,c):
	M = Matrix(r,c,0.0)
	for i in range(r):
		M[i][i] = 1.0
	return M
def m2_zero():
	return Matrix(3, 3, 0.0)
def m2_id():
	return Eye(3, 3)
def m2_rigid(off, a):
	m = Eye(3,3)
	m[0][2] = off[0]
	m[1][2] = off[1]
	c = math.cos(a); s = math.sin(a);
	m[0][0] = c; m[0][1] = -s; m[1][0] = s; m[1][1] = c;
	return m
def m2_mul(m1, m2):
	p = Eye(3, 3)
	for i in range(3):
		for j in range(3):
			p[i][j] = m1[i][0]*m2[0][j]+m1[i][1]*m2[1][j]+m1[i][2]*m2[2][j]
	return p
def m2_mulp(m, v):
	p = [0.0, 0.0]
	p[0] = m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]
	p[1] = m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]
	return p
def m2_mulp_a(m, va):
	return [m2_mulp(m, v) for v in va]
def m2_mulv(m, v):
	p = [0.0, 0.0]
	p[0] = m[0][0]*v[0]+m[0][1]*v[1]
	p[1] = m[1][0]*v[0]+m[1][1]*v[1]
	return p
def m2_inv(m):
	mi = Eye(3, 3)
	mi[0][0]=m[1][1]
	mi[0][1]=m[1][0]
	mi[0][2]=-m[0][2]*m[1][1]-m[1][2]*m[1][0]
	mi[1][0]=m[0][1]
	mi[1][1]=m[0][0]
	mi[1][2]=-m[0][2]*m[0][1]-m[1][2]*m[0][0]
	return mi
def m2_orth(m):
	orth = v2_orth([m[0][0], m[0][1]])
	m[1][0] = orth[0]
	m[1][1] = orth[1]
	return m
def m2_get_trans(m):
	return [m[0][2], m[1][2]]
def m2_get_dir1(m):
	return [m[0][0], m[1][0]]
def m2_set_trans(m, off):
	m[0][2] = off[0]
	m[1][2] = off[1]
def m2_get_angle(m):
	return v2_v_angle([m[0][0], m[1][0]])
def m2_set_angle(m, a):
	c = math.cos(a); s = math.sin(a);
	m[0][0] = c; m[0][1] = -s; m[1][0] = s; m[1][1] = c;
def vec_randomize(v, scl):
	return [x + (0.5*(-1.0 + 2.0*random.random()))*scl for x in v]
def m2_rigid_randomize(pa, rand_settings):
	return m2_rigid(vec_randomize(pa[0], rand_settings[0]), vec_randomize([pa[1]], rand_settings[1])[0])
#
g_mouse = [None, None]
g_buttons = {}
g_drag = {}
g_keys = {}
g_special_keys = {}
g_track = True
g_mouseFocus = True
g_frames = 0
g_fps_frames = 0
g_fps_t0 = 0
g_wh = [1024, 768]; g_wh = [800,600];
g_side_wh = [int(x) for x in v2_muls(g_wh, 0.5)]
def disp_aspect(wh = None):  wh = wh if wh is not None else g_wh; return float(wh[0])/float(wh[1]);
g_zoom = 1.0
g_offset = [0.0, 0.0]
#
def rgb_to_f(rgb):
	return [x/255.0 for x in rgb]
k_white = [1.0, 1.0, 1.0]; k_green = [0.0, 1.0, 0.0]; k_blue = [0.0, 0.0, 1.0];
k_red = [1.0, 0.0, 0.0]; k_lgray = [0.7, 0.7, 0.7]; k_dgray = [0.3, 0.3, 0.3];
k_pink = rgb_to_f([245, 183, 177]); k_bluish1 = rgb_to_f([30, 136, 229]);
k_orange = rgb_to_f([251, 140, 0]);
k_e_1 = [1.0, 0.0]; k_e_2 = [0.0, 1.0]; k_e2 = [k_e_1, k_e_2];
def style0_color3f(r,g,b):
	return (r, g, b)
def style1_color3f(r,g,b):
	return (1.0-r, 1.0-g, 1.0-b)
style_color3f = style0_color3f
def style_glColor3f(r,g,b):
	glColor3f(*style_color3f(r,g,b))
#
def screen_to_draw(pt, wh = None):
	x,y,z = gluUnProject(pt[0], (wh if wh else g_wh)[1] - pt[1], 0);
	return [x,y];
def size_to_draw(sz, wh = None):
	p0 = screen_to_draw(v2_z(), wh); p = screen_to_draw(sz, wh); return v2_sub(p, p0);
#
def draw_strings(strs, x0, y0, col, wh = None, anchor = 'lt', fill_col = None):
	def get_string_height(): return 13
	def get_string_size(str): return [len(str)*8, get_string_height()]
	wh = wh if wh is not None else g_wh
	if (anchor in ['cc', 'ct', 'lb']):
		bounds = [0, 0]
		for str in strs:
			sz = get_string_size(str)
			bounds[0] = m_max(bounds[0], sz[0]); bounds[1] = bounds[1] + sz[1]
		dbounds = size_to_draw(bounds, wh)
		if (anchor[0] == 'c'):
			x0 = x0 - 0.5*dbounds[0]
		if (anchor[1] == 'c'):
			y0 = y0 - 0.5*dbounds[1]
		elif (anchor[1] == 'b'):
			y0 = y0 - dbounds[1]
	style_glColor3f(col[0],col[1],col[2])
	h = size_to_draw([0.0, get_string_height()], wh)[1]
	glPushMatrix();
	glTranslatef(x0, y0+h, 0); glRasterPos2f(0, 0);
	si = 0
	for str in strs:
		for c in str:
			glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(c))
		sz = size_to_draw(get_string_size(str), wh)
		glTranslatef(0, h, 0); glRasterPos2f(0, 0);
		si = si+1
	glPopMatrix()
#
def draw_verts(verts, mode):
	glBegin(mode)
	for p in verts:
		glVertex2f(p[0], p[1])
	glEnd()
def trace_poly(poly, col = k_white):
	style_glColor3f(col[0],col[1],col[2])
	draw_verts(poly, GL_LINE_STRIP)
def fill_poly(poly, col = k_white):
	if len(poly) == 2:
		return trace_poly(poly, col)
	style_glColor3f(col[0],col[1],col[2])
	draw_verts(poly, GL_POLYGON)
def point_poly(poly, col = k_white):
	style_glColor3f(col[0],col[1],col[2])
	glPointSize(3)
	draw_verts(poly, GL_POINTS)
def draw_lines(lines, col = k_white):
	style_glColor3f(col[0],col[1],col[2])
	draw_verts(lines, GL_LINES)
def draw_poly_with_mode(mode, poly, col = k_white):
	if mode == 2:
		fill_poly(poly,col)
	elif mode == 1:
		trace_poly(poly,col)
	else:
		point_poly(poly,col)
#
def make_wh_poly(rwh):
	rw,rh = rwh[0], rwh[1]
	return [ v2_p(-rw, -rh), v2_p(-rw, rh), v2_p(rw, rh), v2_p(rw, -rh), v2_p(-rw, -rh) ]
def make_w_poly(rw):
		return [ v2_p(-rw, 0.0), v2_p(rw, 0.0) ]
def make_h_poly(rh):
		return [ v2_p(0.0, -rh), v2_p(0.0, rh) ]
def make_tri_poly(wh):
	return [v2_p(0,0), v2_p(wh[0], wh[1]), v2_p(wh[0], -wh[1]), v2_p(0,0)]
def draw_stars(stars, col = k_white):
	if len(stars):
		style_glColor3f(col[0],col[1],col[2])
		glBegin(GL_LINES)
		for s in stars:
			for p in s:
				glVertex2f(p[0], p[1])
		glEnd()
def make_star(r, boxr = 0.0):
	star =  [ v2_p(-r, 0), v2_p(r, 0), v2_p(0, -r), v2_p(0, r) ]
	if boxr > 0.0:
		r = boxr
		star.extend( [v2_p(-r, -r), v2_p(-r, r), v2_p(-r, r), v2_p(r, r), v2_p(r, r), v2_p(r, -r), v2_p(r, -r), v2_p(-r, -r) ] )
	return star
#
def makeKeyMods():
	ret = []
	if glutGetModifiers():
		if glutGetModifiers() & 1:
			ret.append('shift')
	return ret
def handleKeys(key, x, y):
	global g_keys
	g_keys[key] = {'wpt':[x,y], 'mod':makeKeyMods() }
def handleSpecialKeys(key, x, y):
	global g_special_keys
	g_special_keys[str(key)] = {'wpt':[x,y], 'mod':makeKeyMods() }
def handleMouseAct(button, mode, x, y):
	global g_buttons
	g_buttons[button] = {'button': button, 'mode':mode, 'wpt':[x,y], 'mod':makeKeyMods() }
	if mode == 0:
		g_drag[button] = g_buttons[button]; g_drag[button]['wpts'] = [g_buttons[button]['wpt']]*3; g_drag[button]['active'] = True;
	elif mode == 1:
		if button in g_drag:
			g_drag[button]['active'] = False
def handleMouseWheel(wheel, direction, x, y):
	print wheel, direction
def handleMousePassiveMove(x, y):
	global g_mouse
	if (g_mouseFocus):
		g_mouse = [x,y]
	for d in [xx for xx in g_drag.values() if xx['active']]:
		# wpts: src, curr, prev
		d['wpts'][2] = d['wpts'][1]
		d['wpts'][1] = [x,y]
def handleMouseMove(x, y):
	handleMousePassiveMove(x, y)
def handleMouseEntry(state):
	global g_mouseFocus
	g_mouseFocus = (state == GLUT_ENTERED)
def dbg_mouse():
	for d in [x for x in g_drag.values() if x['active']]:
		print '({},{}) -> ({},{})'.format(d['wpts'][0][0], d['wpts'][0][1], d['wpts'][1][0], d['wpts'][1][1])
def handleReshape(w, h):
	global g_wh
	g_wh = [w,h]
	glutDisplayFunc(lambda: display(w, h))
	glutPostRedisplay()
def main_realtime():
	glutInit(sys.argv)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
	glutInitWindowSize(g_wh[0], g_wh[1])
	glutCreateWindow('relearn')
	glutReshapeFunc(handleReshape)
	#
	glutIdleFunc(glutPostRedisplay)
	glutMouseFunc(handleMouseAct)
	glutPassiveMotionFunc(handleMousePassiveMove)
	glutMotionFunc(handleMouseMove)
	glutEntryFunc(handleMouseEntry)
	glutKeyboardFunc(handleKeys)
	glutSpecialFunc(handleSpecialKeys)
	#
	glutMainLoop()
def display(w, h):
	def start_display(w, h):
		global g_frames
		aspect = disp_aspect([w,h])
		glViewport(0, 0, w, h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(-1*aspect*g_zoom+g_offset[0], 1*aspect*g_zoom+g_offset[0], -1*g_zoom+g_offset[1], 1*g_zoom+g_offset[1], -1, 1)
		#
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity()
		#
		(r,g,b) = style_color3f(0,0,0)
		glClearColor(r,g,b, 0.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		#
		g_frames = g_frames+1
	def end_display():
		global g_fps_frames, g_fps_t0
		glutSwapBuffers()
		t1 = glutGet(GLUT_ELAPSED_TIME)
		g_fps_frames = g_fps_frames+1
		if (g_fps_t0 <= 0) or (t1-g_fps_t0 >= 1000.0):
			g_fps = (1000.0 * float(g_fps_frames)) / float(t1-g_fps_t0)
			g_fps_t0 = t1; g_fps_frames = 0;
		#print (t1-t0)
	start_display(w, h)
	do_display()
	end_display()
def aabb_inflate(vol, rad):
	return [vol[0]-rad, vol[1]-rad, vol[2]+rad, vol[3]+rad]
def aabb_size(vol):
	# min_1, min_2, max_1, max_2
	return  m_max(vol[2]-vol[0], 0.0) * m_max(vol[3]-vol[1], 0.0)
def aabb_radius(vol):
	return m_max(m_max(vol[2]-vol[0], 0.0), m_max(vol[3]-vol[1], 0.0))
def aabb_mM(vol):
	return [vol[:2], vol[2:]]
def aabb_equals(vol1, vol2):
	return all([vol1[i] == vol2[i] for i in range(4)])
def aabb_inter(vol1, vol2):
	return [ m_max(vol1[0], vol2[0]), m_max(vol1[1], vol2[1]), m_min(vol1[2], vol2[2]), m_min(vol1[3], vol2[3])  ]
def aabb_union(vol1, vol2):
	return [ m_min(vol1[0], vol2[0]), m_min(vol1[1], vol2[1]), m_max(vol1[2], vol2[2]), m_max(vol1[3], vol2[3])  ]
def make_aabb_btree():
	return {'func_vol_size':aabb_size, 'func_vol_inter':aabb_inter, 'func_vol_union':aabb_union, 'root':None}
def point_to_aabb(pt):
	return [pt[0], pt[1], pt[0], pt[1]]
def points_to_aabb(pts):
	return reduce(lambda x, y: aabb_union(x,y), [point_to_aabb(x) for x in pts])
def btree_intersecting_leaves(tree, volume):
	def btree_intersects(func_vol_size, func_vol_inter, node, vol, inter_leaves):
		if func_vol_size(func_vol_inter(node['vol'], vol)) > 0:
			if node['is_leaf']:
				inter_leaves.append(node)
			else:
				btree_intersects(func_vol_size, func_vol_inter, node['l'], vol, inter_leaves)
				btree_intersects(func_vol_size, func_vol_inter, node['r'], vol, inter_leaves)
		else:
			return
	inter_leaves = []
	if tree['root'] is not None:
		btree_intersects(tree['func_vol_size'], tree['func_vol_inter'], tree['root'], volume, inter_leaves)
	return inter_leaves
def btree_insert(tree, volume, data = None):
	if tree['root'] is None:
		tree['root'] = { 'is_leaf':True, 'vol':volume, 'data':data, 'p':None }
	else:
		inter_leaves = btree_intersecting_leaves(tree, volume)
		if len(inter_leaves) == 0:
			lr_vol = tree['func_vol_union']( tree['root']['vol'], volume )
			new_l = { 'is_leaf':True, 'vol':volume, 'data':data }
			new_r = tree['root'];
			new_root = { 'is_leaf':False, 'vol':lr_vol, 'l':new_l, 'r':new_r, 'p':None }
			new_l['p'] = new_root; new_r['p'] = new_root;
			tree['root'] = new_root
		else:
			func_vol_size = tree['func_vol_size']; func_vol_inter = tree['func_vol_inter'];
			vol_sizes = [ func_vol_size(func_vol_inter(x['vol'], volume)) for x in inter_leaves ]
			by_size = sorted( range(len(vol_sizes)), key = lambda x: vol_sizes[x] )
			split_leaf = inter_leaves[by_size[0]]
			lr_vol = tree['func_vol_union']( split_leaf['vol'], volume )
			new_l = { 'is_leaf':True, 'vol':volume, 'data':data, 'p':split_leaf }
			new_r = { 'is_leaf':True, 'vol':split_leaf['vol'], 'data':split_leaf['data'], 'p':split_leaf }
			split_leaf['is_leaf'] = False; split_leaf['vol'] = lr_vol;
			split_leaf['l'] = new_l; split_leaf['r'] = new_r;
			walk_c = split_leaf; func_vol_union = tree['func_vol_union'];
			while walk_c['p'] is not None:
				walk_c['p']['vol'] = func_vol_union(walk_c['p']['vol'], walk_c['vol'])
				walk_c = walk_c['p']
def btree_size(tree):
	def btree_node_size(node):
		return 1 + 0 if node['is_leaf'] else (btree_node_size(node['l']) + btree_node_size(node['r']))
	return btree_node_size(tree['root']) if tree['root'] is not None else 0
#
g_scene = None
def do_display():
	global g_scene
	if g_scene is None:
		settings = { }
		g_scene = {}
		g_scene['scene_offset'] = { 'active':False }
		g_scene['zoom_offset'] = { 'active':False }
		g_scene['mouse'] = { 'track_pts':[], 'max_track_pts':140, 'tracking':False }
		#g_scene['design'] = { 'active_drag':False, 'active_drag_body_col':k_pink, 'static_col':k_blue }
		g_scene['show'] = { }
		g_scene['show_keys'] = { }
		g_scene['on'] = { }
		g_scene['on_keys'] = { }
		g_scene['once'] = {  }
		g_scene['once_keys'] = {  }
		g_scene['scroll'] = { 'modes':['zoom'], 'mode':0 }
	scene = g_scene
	#
	if ('\x1b' in g_keys.keys()):
		if glutLeaveMainLoop:
			glutLeaveMainLoop()
		sys.exit(0)
		return
	if ('\t' in g_keys.keys()):
		scene['scroll']['mode'] = (scene['scroll']['mode']+1) % len(scene['scroll']['modes'])
	for coll, coll_keys in [['show', 'show_keys'], ['on', 'on_keys'], ['once', 'once_keys']]:
		for sk in g_special_keys.keys():
			name_key = g_scene[coll_keys].get(sk, '')
			if name_key != '':
				scene[coll][name_key] = not scene[coll][name_key]
	#
	def track_mouse():
		if 't' in g_keys:
			scene['mouse']['tracking'] = not scene['mouse']['tracking']
		track =	scene['mouse']['track_pts']
		if scene['mouse']['tracking']:
			if g_mouseFocus and g_mouse[0] is not None:
				if v2_dist(scene.get('mouse_last', g_mouse), g_mouse) > 3.0 * k_cm:
					track.append(screen_to_draw(g_mouse))
				if len(track) > scene['mouse']['max_track_pts']:
					scene['mouse']['track_pts'] = track[1:]
				scene['mouse_last'] = g_mouse
				trace_poly( m2_mulp_a(m2_rigid(screen_to_draw(g_mouse), 0), make_wh_poly([k_cm*5, k_cm*5])) )
		else:
			if len(track) >= 2:
				scene['rig']['path'] = copy.copy(track)
				del track[:]
		if len(track) >= 2:
			trace_poly(track, [1.0, 1.0, 0.0])
	def handle_scene_zoom():
		global g_zoom
		zd = 1.0
		if any([x in g_keys for x in ['-','=']]):
			zd = 3.0/4.0 if '-' in g_keys else (4.0/3.0 if '=' in g_keys else 1.0)
			g_zoom = m_min(m_max(g_zoom * zd, 0.001), 100)
	def handle_input_scroll():
		def do_generic_scroll(precision, zmin, zmax, scroll_key):
			zd = precision if 3 in g_buttons else (1.0/precision if 4 in g_buttons else 1.0)
			if scroll_key not in scene['scroll']:
				scene['scroll'][scroll_key] = 1.0
			zoom = scene['scroll'][scroll_key]; zoom = m_min(m_max(zoom * zd, zmin), zmax);
			scene['scroll'][scroll_key] = zoom; scene['scroll']['value'] = '{0:.0f}%'.format(zoom*100);
			return zoom
		scroll_mode = scene['scroll']['modes'][scene['scroll']['mode']]
		scene['scroll']['value'] = ''
		if scroll_mode == 'zoom':
			handle_scene_zoom()
			scene['scroll']['value'] = '{0:.0f}%'.format(1.0/g_zoom*100)
	def handle_draw_strings():
		lines = []
		lines.append( 'scroll: [{}:{}]'.format(scene['scroll']['modes'][scene['scroll']['mode']], scene['scroll'].get('value','') ) )
		draw_strings(lines, -1.0*g_zoom*disp_aspect()+g_offset[0], 1.0*g_zoom+g_offset[1], k_white)
	def handle_scene_offset():
		global g_offset
		drag = g_drag.get(1, None)
		if drag is not None and drag['active']:
			if scene['scene_offset']['active'] == False:
				scene['scene_offset']['orig_offset'] = g_offset
				scene['scene_offset']['active'] = True
			pts = [screen_to_draw(x) for x in drag['wpts']]
			g_offset = v2_add(scene['scene_offset']['orig_offset'], v2_sub(pts[0], pts[1]))
		else:
			scene['scene_offset']['active'] = False
		if 'r' in g_keys:
			g_offset = v2_z()	#
	def draw_design_grid(unit):
		draw_lines([v2_z(), v2_muls(k_e_1, unit), v2_z(), v2_muls(k_e_2, unit)], k_lgray)
	draw_design_grid(1.0/5.0)
	track_mouse()
	if g_dbg:
		dbg_mouse()
	handle_draw_strings()
	#
	for once_k in scene['once'].keys():
		scene['once'][once_k] = False
	#
	handle_input_scroll()
	g_keys.clear(); g_special_keys.clear(); g_buttons.clear();
def main():
	main_realtime()
#
main()
