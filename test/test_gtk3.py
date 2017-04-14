import sys
import os
import gtk
import math
import cairo
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
def load_lrn_file(fpath):
	data = {'pos':None, 'scale':1.0}
	with open(fpath) as fi:
		lhead = None
		for line in fi.readlines():
			line = line.strip()
			if lhead:
				if lhead == 'pos.x':
					data['pos'] = [0,0]
					data['pos'][0] = float(line)*2.0
				elif lhead == 'pos.y':
					data['pos'][1] = float(line)*2.0
				elif lhead == 'scl':
					data['scale'] = float(line)
				lhead = None
			else:
				lhead = line
	return data
def load_media_file(fpath, lf_data):
	data = { 'type':'', 'lf':lf_data }
	if os.path.splitext(fpath)[1] == '.png':
		data['type'] = 'png'
		data['fpath'] = fpath
	return data
def update_media_texture(scene, md):
	if md['type'] == 'png':
		if 'surf' not in md:
			md['surf'] = {}
			if scene['dbg_fs']:
				sys.stdout.write('Loading [{}]..'.format(md['fpath'])); sys.stdout.flush();
			if True:
				md['surf']['surf'] = cairo.ImageSurface.create_from_png(md['fpath'])
				#md['tex']['wh'] = owh
			if scene['dbg_fs']:
				print '.'
		else:
			pass
def scene_load(scene):
	'''def init_voltree():
		scene['voltree'] = make_aabb_btree()
		for md in scene['media_data'].values():
			if md['lf'] and md.get('tex', {}).get('wh', None) is not None:
				pts = [v2_add(md['lf']['pos'], v2_muls(md['tex']['wh'], md['lf']['scale'] * x)) for x in [-0.5, 0.5]]
				md['lf']['vol'] = points_to_aabb(pts)
				btree_insert(scene['voltree'], md['lf']['vol'], md)'''
	if 'path' not in scene:
		path = sys_argv_get(['-path'], '')
		lrn_path = os.path.join(path, 'state')
		if len(path):
			path = os.path.expanduser(path)
			scene['path'] = path; scene['lrn_path'] = lrn_path;
			media_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
			lrn_files = [f for f in os.listdir(lrn_path) if os.path.isfile(os.path.join(lrn_path, f)) and os.path.splitext(f)[1] == '.lrn']
			#print media_files, lrn_files
			scene['lrn_data'] = {}
			for lf in lrn_files:
				lf_data = load_lrn_file(os.path.join(lrn_path, lf))
				scene['lrn_data'][lf] = lf_data
			scene['media_data'] = {}
			for mf in media_files:
				mf_lf = '_'+mf+'.lrn'
				lf_data = scene['lrn_data'].get(mf_lf, None)
				mf_data = load_media_file(os.path.join(path, mf), lf_data)
				scene['media_data'][mf] = mf_data
				update_media_texture(scene, mf_data)
		else:
			scene['path'] = path
def display_scene(scene, cr, wh):
	def display_md(md, scl, wh):
		if md['type'] == 'png' and md['lf']:
			pos = md['lf']['pos']
			cr.save()
			cr.translate(wh[0]/2+pos[0]*scl, wh[1] - (wh[1]/2+pos[1]*scl))
			cr.scale(scl, scl)
			cr.set_source_surface(md['surf']['surf'], 0, 0)
			cr.paint()
			cr.restore()
	scl = 1.0 / scene['zoom']
	for md in scene['media_data'].values():
		display_md(md, scl, wh)
def make_scene():
	return {'dbg_fs':sys_argv_has(['-dbg_fs']), 'zoom':10.0}

class PyApp(gtk.Window):
		def __init__(self):
				super(PyApp, self).__init__()

				self.scene = make_scene()
				scene_load(self.scene)

				self.set_title("relearn")
				self.resize(800, 600)
				self.set_position(gtk.WIN_POS_CENTER)

				self.connect('destroy', gtk.main_quit)
				accelgroup = gtk.AccelGroup()
				key, modifier = gtk.accelerator_parse('Escape')
				accelgroup.connect_group(key,modifier,gtk.ACCEL_VISIBLE,gtk.main_quit)
				self.add_accel_group(accelgroup)

				darea = gtk.DrawingArea()
				darea.connect("expose-event", self.expose)
				self.add(darea)

				darea.connect("button_press_event", self.button_press_event)
				self.connect("key_press_event", self.key_press_event)
				darea.set_events(gtk.gdk.EXPOSURE_MASK|gtk.gdk.LEAVE_NOTIFY_MASK|
					gtk.gdk.BUTTON_PRESS_MASK | gtk.gdk.POINTER_MOTION_MASK| gtk.gdk.POINTER_MOTION_HINT_MASK|
					gtk.gdk.KEY_PRESS_MASK)

				self.show_all()
		def button_press_event(self, widget, event):
			if event.button == 1:
				self.queue_draw()
			return True
		def key_press_event(self, widget, event):
			if event.keyval == gtk.keysyms.equal:
				self.scene['zoom'] = 3.0/4.0 * self.scene['zoom']
				self.queue_draw()
			elif event.keyval == gtk.keysyms.minus:
				self.scene['zoom'] = 4.0/3.0 * self.scene['zoom']
				self.queue_draw()
			return True
		def expose(self, widget, event):
				cr = widget.window.cairo_create()
				wh = [self.allocation.width, self.allocation.height]
				cr.set_source_rgba(1, 1, 1, 1); cr.rectangle(0, 0, wh[0], wh[1]); cr.fill();
				display_scene(self.scene, cr, wh)
#
PyApp()
gtk.main()