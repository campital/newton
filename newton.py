from scene import *
import numpy as np
import ui

fractal_shader_code = '''
precision highp float;
varying vec2 v_tex_coord;
uniform int max_trials;
uniform vec2 size;
uniform vec2 offset;

uniform vec2 polynomial_x;
uniform vec2 polynomial_y;
uniform vec2 polynomial_z;
uniform vec2 polynomial_w;

uniform vec2 derivative_x;
uniform vec2 derivative_y;
uniform vec2 derivative_z;

uniform vec2 root_a;
uniform vec2 root_b;
uniform vec2 root_c;

vec2 mul_complex(vec2 a, vec2 b) {
	return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 poly(vec2 x) {
	return polynomial_x + mul_complex(polynomial_y, x) + mul_complex(polynomial_z, mul_complex(x, x)) + mul_complex(polynomial_w, mul_complex(x, mul_complex(x, x)));
}

vec2 deriv(vec2 x) {
	return derivative_x + mul_complex(derivative_y, x) + mul_complex(derivative_z, mul_complex(x, x));
}

vec4 newton(vec2 pos) {
	for(int i = 0; i < max_trials; i++) {
		vec2 d = deriv(pos);
		vec2 v = poly(pos);
		pos = pos - mul_complex(v, vec2(d.x, -d.y)) / (d.x * d.x + d.y * d.y);
		if(pow(pos.x - root_a.x, 2.0) + pow(pos.y - root_a.y, 2.0) < 0.001) {
			return vec4(1.0, 0.0, 0.0, 1.0);
		}
		if(pow(pos.x - root_b.x, 2.0) + pow(pos.y - root_b.y, 2.0) < 0.001) {
			return vec4(0.0, 1.0, 0.0, 1.0);
		}
		if(pow(pos.x - root_c.x, 2.0) + pow(pos.y - root_c.y, 2.0) < 0.001) {
			return vec4(0.0, 0.0, 1.0, 1.0);
		}
	}
	return vec4(0.0, 0.0, 0.0, 1.0);
}

void main() {
	gl_FragColor = newton((v_tex_coord - vec2(0.5, 0.5)) * size + offset);
}
'''

class Fractal (Scene):
	def setup(self):
		self.fractal_renderer = SpriteNode(None, size=self.size, parent=self)
		self.fractal_renderer.anchor_point = (0.0, 0.0)
		self.fractal_shader = Shader(fractal_shader_code)
		self.fractal_shader.set_uniform('max_trials', 40)
		self.fractal_renderer.shader = self.fractal_shader
		
		self.touched = None
		self.panning = False
		handle_path = ui.Path.oval(0, 0, 30, 30)
		handle_path.line_width = 2
		self.root_controls = []
		for i in range(3):
			self.root_controls.append(ShapeNode(handle_path, (0.0, 0.0, 0.0, 0.4), 'black', parent=self))
		
		self.function_label = LabelNode('', position=(10.0, self.size[1] - 10.0), parent=self)
		self.function_label.anchor_point = (0.0, 1.0)
		self.function_label.color = (0.0, 0.0, 0.0)
		
		self.reset_button = ButtonNode('Reset', position=(self.size[0] - 10.0, 10.0), parent=self)
		self.reset_button.anchor_point = (1.0, 0.0)
		self.reset_button.title_label.position = (-self.reset_button.size[0] / 2, self.reset_button.size[1] / 2)
		self.reset_button.pressed = False
		self.reset()

	def update(self):
		pass

	def did_change_size(self):
		self.fractal_renderer.size = self.size
		height = self.desired_size[1]
		if self.size[0] / self.size[1] < self.desired_size[0] / self.desired_size[1]:
			height = height * (self.desired_size[0] / (self.size[0] * height / self.size[1]))
		self.fractal_size = (self.size[0] * height / self.size[1], height)
		self.fractal_shader.set_uniform('size', self.fractal_size)
		self.fractal_shader.set_uniform('offset', self.offset)
		for i in range(3):
			self.root_controls[i].position = self.poly_to_screen((np.real(self.roots[i]), np.imag(self.roots[i])))
		self.function_label.position = (10.0, self.size[1] - 10.0)
		self.reset_button.position = position=(self.size[0] - 10.0, 10.0)

	def touch_began(self, touch):
		if touch.location in self.reset_button.frame:
			self.reset_button.texture = Texture('pzl:Button2')
			self.reset_button.pressed = True
			self.reset_button.touch_id = touch.touch_id
		elif len(self.touches) == 1:
			for i, control in enumerate(self.root_controls):
				if touch.location in control.frame:
					self.touched = i
		if self.touched == None and not self.reset_button.pressed:
			self.pan_start = self.screen_to_poly(self.touch_average())
			self.pan_start_offset = self.offset
			self.panning = True
			if len(self.touches) == 2:
				self.zoom_size = self.touch_distance()
				self.zoom_original = self.desired_size
	
	def touch_moved(self, touch):
		if self.touched != None:
			self.root_controls[self.touched].position = touch.location
			self.prepare_polynomial()
		elif self.panning:
			pan_new = self.screen_to_poly(self.touch_average(), self.pan_start_offset)
			self.offset = (self.pan_start_offset[0] - pan_new[0] + self.pan_start[0], self.pan_start_offset[1] - pan_new[1] + self.pan_start[1])
			if len(self.touches) == 2:
				zoom_scalar = self.zoom_size / self.touch_distance()
				self.desired_size = (self.zoom_original[0] * zoom_scalar, self.zoom_original[1] * zoom_scalar)
			self.did_change_size()
			

	def touch_ended(self, touch):
		self.touched = None
		self.panning = False
		if len(self.touches) > 0:
			self.pan_start = self.screen_to_poly(self.touch_average())
			self.pan_start_offset = self.offset
			if len(self.touches) == 2:
				self.zoom_original = self.desired_size
				self.zoom_size = self.touch_distance()
		if self.reset_button.pressed and (touch.location in self.reset_button.frame or touch.touch_id == self.reset_button.touch_id):
			self.reset_button.texture = Texture('pzl:Button1')
			self.reset_button.pressed = False
			if touch.location in self.reset_button.frame and touch.touch_id == self.reset_button.touch_id:
				self.reset()

	def prepare_polynomial(self):
		self.roots = []
		for x in self.root_controls:
			pos = self.screen_to_poly(x.position)
			self.roots.append(complex(pos[0], pos[1]))
		poly = np.polynomial.polynomial.polyfromroots(self.roots)
		self.fractal_shader.set_uniform('polynomial_x', (np.real(poly[0]), np.imag(poly[0])))
		self.fractal_shader.set_uniform('polynomial_y', (np.real(poly[1]), np.imag(poly[1])))
		self.fractal_shader.set_uniform('polynomial_z', (np.real(poly[2]), np.imag(poly[2])))
		self.fractal_shader.set_uniform('polynomial_w', (np.real(poly[3]), np.imag(poly[3])))
		deriv = np.polynomial.polynomial.polyder(poly)
		self.fractal_shader.set_uniform('derivative_x', (np.real(deriv[0]), np.imag(deriv[0])))
		self.fractal_shader.set_uniform('derivative_y', (np.real(deriv[1]), np.imag(deriv[1])))
		self.fractal_shader.set_uniform('derivative_z', (np.real(deriv[2]), np.imag(deriv[2])))
		self.fractal_shader.set_uniform("root_a", (np.real(self.roots[0]), np.imag(self.roots[0])))
		self.fractal_shader.set_uniform("root_b", (np.real(self.roots[1]), np.imag(self.roots[1])))
		self.fractal_shader.set_uniform("root_c", (np.real(self.roots[2]), np.imag(self.roots[2])))
		self.function_label.text = f'f(x)={round(poly[3], 2)}x^3+{round(poly[2], 2)}x^2+{round(poly[1], 2)}x+{round(poly[0], 2)}'
		
	def reset(self):
		self.roots = [complex(1.0, 0.0), complex(-0.5, np.sqrt(3) / 2), complex(-0.5, -np.sqrt(3) / 2)]
		self.desired_size = (3.0, 2.0)
		self.offset = (0.0, 0.0)
		self.did_change_size()
		self.prepare_polynomial()

	def poly_to_screen(self, coord):
		return ((coord[0] - self.offset[0] + self.fractal_size[0] / 2) * self.size[0] / self.fractal_size[0], (coord[1] - self.offset[1] + self.fractal_size[1] / 2) * self.size[1] / self.fractal_size[1])

	def touch_average(self):
		x_total = 0
		y_total = 0
		i = 0
		for k, x in self.touches.items():
			i += 1
			x_total += x.location[0]
			y_total += x.location[1]
		return (x_total / i, y_total / i)

	def touch_distance(self):
		touches = list(self.touches.values())
		return np.sqrt(pow(touches[0].location[0] - touches[1].location[0], 2) + pow(touches[0].location[1] - touches[1].location[1], 2))

	def screen_to_poly(self, coord, offset=None):
		if offset == None:
			offset = self.offset
		return ((coord[0] / self.size[0]) * self.fractal_size[0] - self.fractal_size[0] / 2 + offset[0], (coord[1] / self.size[1]) * self.fractal_size[1] - self.fractal_size[1] / 2 + offset[1])

class ButtonNode (SpriteNode):
	def __init__(self, title, *args, **kwargs):
		SpriteNode.__init__(self, 'pzl:Button1', *args, **kwargs)
		self.title_label = LabelNode(title, color='black', parent=self)
		self.title = title

if __name__ == '__main__':
	run(Fractal(), show_fps=False)