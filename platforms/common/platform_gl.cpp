#include "platform_gl.h"

// enable GL error checking
//#define DEBUG 1

#include "gl.h"
#include "gl/glError.h"

namespace Tangram {

GLenum GL::getError() {
    return glGetError();
}

const GLubyte* GL::getString(GLenum name) {
#ifdef GL_VERSION_3_0
    // technically, we should do glGetStringi instead
    if(name == GL_EXTENSIONS) return (const GLubyte*)"vertex_array_object texture_non_power_of_two";
#endif
    auto result = glGetString(name);
    GL_CHECK({});
    return result;
}

void GL::clear(GLbitfield mask) {
    GL_CHECK(glClear(mask));
}
void GL::lineWidth(GLfloat width) {
    GL_CHECK(glLineWidth(width));
}
void GL::viewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    GL_CHECK(glViewport(x, y, width, height));
}

void GL::enable(GLenum id) {
    GL_CHECK(glEnable(id));
}
void GL::disable(GLenum id) {
    GL_CHECK(glDisable(id));
}
void GL::depthFunc(GLenum func) {
    GL_CHECK(glDepthFunc(func));
}
void GL::depthMask(GLboolean flag) {
    GL_CHECK(glDepthMask(flag));
}
void GL::depthRange(GLfloat n, GLfloat f) {
    GL_CHECK(glDepthRangef(n, f));
}
void GL::clearDepth(GLfloat d) {
    GL_CHECK(glClearDepthf(d));
}
void GL::blendFunc(GLenum sfactor, GLenum dfactor) {
    GL_CHECK(glBlendFunc(sfactor, dfactor));
}
void GL::stencilFunc(GLenum func, GLint ref, GLuint mask) {
    GL_CHECK(glStencilFunc(func, ref, mask));
}
void GL::stencilMask(GLuint mask) {
    GL_CHECK(glStencilMask(mask));
}
void GL::stencilOp(GLenum fail, GLenum zfail, GLenum zpass) {
    GL_CHECK(glStencilOp(fail, zfail, zpass));
}
void GL::clearStencil(GLint s) {
    GL_CHECK(glClearStencil(s));
}
void GL::colorMask(GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha) {
    GL_CHECK(glColorMask(red, green, blue, alpha));
}
void GL::cullFace(GLenum mode) {
    GL_CHECK(glCullFace(mode));
}
void GL::frontFace(GLenum mode) {
    GL_CHECK(glFrontFace(mode));
}
void GL::clearColor(GLclampf red, GLclampf green, GLclampf blue, GLclampf alpha) {
    GL_CHECK(glClearColor(red, green, blue, alpha));
}
void GL::getIntegerv(GLenum pname, GLint *params ) {
    GL_CHECK(glGetIntegerv(pname, params ));
}

// Program
void GL::useProgram(GLuint program) {
    GL_CHECK(glUseProgram(program));
}
void GL::deleteProgram(GLuint program) {
    GL_CHECK(glDeleteProgram(program));
}
void GL::deleteShader(GLuint shader) {
    GL_CHECK(glDeleteShader(shader));
}
GLuint GL::createShader(GLenum type) {
    auto result = glCreateShader(type);
    GL_CHECK({});
    return result;
}
GLuint GL::createProgram() {
    auto result = glCreateProgram();
    GL_CHECK({});
    return result;
}

void GL::compileShader(GLuint shader) {
    GL_CHECK(glCompileShader(shader));
}
void GL::attachShader(GLuint program, GLuint shader) {
    GL_CHECK(glAttachShader(program,shader));
}
void GL::linkProgram(GLuint program) {
    GL_CHECK(glLinkProgram(program));
}

void GL::shaderSource(GLuint shader, GLsizei count, const GLchar **string, const GLint *length) {
        auto source = const_cast<const GLchar**>(string);
    GL_CHECK(glShaderSource(shader, count, source, length));
}
void GL::getShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    GL_CHECK(glGetShaderInfoLog(shader, bufSize, length, infoLog));
}
void GL::getProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog) {
    GL_CHECK(glGetProgramInfoLog(program, bufSize, length, infoLog));
}
GLint GL::getUniformLocation(GLuint program, const GLchar *name) {
    auto result = glGetUniformLocation(program, name);
    GL_CHECK({});
    return result;
}
GLint GL::getAttribLocation(GLuint program, const GLchar *name) {
    auto result = glGetAttribLocation(program, name);
    GL_CHECK({});
    return result;
}
void GL::bindAttribLocation(GLuint program, GLuint index, const GLchar *name) {
    GL_CHECK(glBindAttribLocation(program, index, name));
}
void GL::getProgramiv(GLuint program, GLenum pname, GLint *params) {
    GL_CHECK(glGetProgramiv(program,pname,params));
}
void GL::getShaderiv(GLuint shader, GLenum pname, GLint *params) {
    GL_CHECK(glGetShaderiv(shader,pname, params));
}

// Buffers
void GL::bindBuffer(GLenum target, GLuint buffer) {
    GL_CHECK(glBindBuffer(target, buffer));
}
void GL::deleteBuffers(GLsizei n, const GLuint *buffers) {
    GL_CHECK(glDeleteBuffers(n, buffers));
}
void GL::genBuffers(GLsizei n, GLuint *buffers) {
    GL_CHECK(glGenBuffers(n, buffers));
}
void GL::bufferData(GLenum target, GLsizeiptr size, const void *data, GLenum usage) {
    GL_CHECK(glBufferData(target, size, data, usage));
}
void GL::bufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void *data) {
    GL_CHECK(glBufferSubData(target, offset, size, data));
}
void GL::readPixels(GLint x, GLint y, GLsizei width, GLsizei height,
                    GLenum format, GLenum type, GLvoid* pixels) {
    GL_CHECK(glReadPixels(x, y, width, height, format, type, pixels));
}

// Texture
void GL::bindTexture(GLenum target, GLuint texture ) {
    GL_CHECK(glBindTexture(target, texture ));
}
void GL::activeTexture(GLenum texture) {
    GL_CHECK(glActiveTexture(texture));
}
void GL::genTextures(GLsizei n, GLuint *textures ) {
    GL_CHECK(glGenTextures(n, textures ));
}
void GL::deleteTextures(GLsizei n, const GLuint *textures) {
    GL_CHECK(glDeleteTextures(n, textures));
}
void GL::texParameteri(GLenum target, GLenum pname, GLint param ) {
    GL_CHECK(glTexParameteri(target, pname, param ));
}
void GL::texImage2D(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height,
                    GLint border, GLenum format, GLenum type, const GLvoid *pixels) {
    GL_CHECK(glTexImage2D(target, level, internalFormat, width, height, border, format, type, pixels)); }

void GL::texSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                       GLenum format, GLenum type, const GLvoid *pixels) {
    GL_CHECK(glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, pixels)); }

void GL::generateMipmap(GLenum target) {
    GL_CHECK(glGenerateMipmap(target));
}

void GL::enableVertexAttribArray(GLuint index) {
    GL_CHECK(glEnableVertexAttribArray(index));
}
void GL::disableVertexAttribArray(GLuint index) {
    GL_CHECK(glDisableVertexAttribArray(index));
}
void GL::vertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized,
                             GLsizei stride, const void *pointer) {
    GL_CHECK(glVertexAttribPointer(index, size, type, normalized, stride, pointer));
}

void GL::drawArrays(GLenum mode, GLint first, GLsizei count ) {
    GL_CHECK(glDrawArrays(mode, first, count ));
}
void GL::drawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid *indices ) {
    GL_CHECK(glDrawElements(mode, count, type, indices ));
}

void GL::uniform1f(GLint location, GLfloat v0) {
    GL_CHECK(glUniform1f(location, v0));
}
void GL::uniform2f(GLint location, GLfloat v0, GLfloat v1) {
    GL_CHECK(glUniform2f(location, v0, v1));
}
void GL::uniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2) {
    GL_CHECK(glUniform3f(location, v0, v1, v2));
}
void GL::uniform4f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3) {
    GL_CHECK(glUniform4f(location, v0, v1, v2, v3));
}

void GL::uniform1i(GLint location, GLint v0) {
    GL_CHECK(glUniform1i(location, v0));
}
void GL::uniform2i(GLint location, GLint v0, GLint v1) {
    GL_CHECK(glUniform2i(location, v0, v1));
}
void GL::uniform3i(GLint location, GLint v0, GLint v1, GLint v2) {
    GL_CHECK(glUniform3i(location, v0, v1, v2));
}
void GL::uniform4i(GLint location, GLint v0, GLint v1, GLint v2, GLint v3) {
    GL_CHECK(glUniform4i(location, v0, v1, v2, v3));
}

void GL::uniform1fv(GLint location, GLsizei count, const GLfloat *value) {
    GL_CHECK(glUniform1fv(location, count, value));
}
void GL::uniform2fv(GLint location, GLsizei count, const GLfloat *value) {
    GL_CHECK(glUniform2fv(location, count, value));
}
void GL::uniform3fv(GLint location, GLsizei count, const GLfloat *value) {
    GL_CHECK(glUniform3fv(location, count, value));
}
void GL::uniform4fv(GLint location, GLsizei count, const GLfloat *value) {
    GL_CHECK(glUniform4fv(location, count, value));
}
void GL::uniform1iv(GLint location, GLsizei count, const GLint *value) {
    GL_CHECK(glUniform1iv(location, count, value));
}
void GL::uniform2iv(GLint location, GLsizei count, const GLint *value) {
    GL_CHECK(glUniform2iv(location, count, value));
}
void GL::uniform3iv(GLint location, GLsizei count, const GLint *value) {
    GL_CHECK(glUniform3iv(location, count, value));
}
void GL::uniform4iv(GLint location, GLsizei count, const GLint *value) {
    GL_CHECK(glUniform4iv(location, count, value));
}

void GL::uniformMatrix2fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    GL_CHECK(glUniformMatrix2fv(location, count, transpose, value));
}
void GL::uniformMatrix3fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    GL_CHECK(glUniformMatrix3fv(location, count, transpose, value));
}
void GL::uniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value) {
    GL_CHECK(glUniformMatrix4fv(location, count, transpose, value));
}

// mapbuffer
void* GL::mapBuffer(GLenum target, GLenum access) {
    auto result =  glMapBuffer(target, access);
    GL_CHECK({});
    return result;
}
GLboolean GL::unmapBuffer(GLenum target) {
    auto result = glUnmapBuffer(target);
    GL_CHECK({});
    return result;
}

void GL::finish(void) {
    GL_CHECK(glFinish());
}

// VAO
void GL::bindVertexArray(GLuint array) {
    GL_CHECK(glBindVertexArray(array));
}
void GL::deleteVertexArrays(GLsizei n, const GLuint *arrays) {
    GL_CHECK(glDeleteVertexArrays(n, arrays));
}
void GL::genVertexArrays(GLsizei n, GLuint *arrays) {
    GL_CHECK(glGenVertexArrays(n, arrays));
}

// Framebuffer
void GL::bindFramebuffer(GLenum target, GLuint framebuffer) {
    GL_CHECK(glBindFramebuffer(target, framebuffer));
}

void GL::genFramebuffers(GLsizei n, GLuint *framebuffers) {
    GL_CHECK(glGenFramebuffers(n, framebuffers));
}

void GL::framebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget,
                              GLuint texture, GLint level) {
    GL_CHECK(glFramebufferTexture2D(target, attachment, textarget, texture, level));
}

void GL::renderbufferStorage(GLenum target, GLenum internalformat, GLsizei width,
                             GLsizei height) {
    GL_CHECK(glRenderbufferStorage(target, internalformat, width, height));
}

void GL::framebufferRenderbuffer(GLenum target, GLenum attachment,
                                 GLenum renderbuffertarget, GLuint renderbuffer) {
    GL_CHECK(glFramebufferRenderbuffer(target, attachment, renderbuffertarget, renderbuffer));
}

void GL::genRenderbuffers(GLsizei n, GLuint *renderbuffers) {
    GL_CHECK(glGenRenderbuffers(n, renderbuffers));
}

void GL::bindRenderbuffer(GLenum target, GLuint renderbuffer) {
    GL_CHECK(glBindRenderbuffer(target, renderbuffer));
}

void GL::deleteFramebuffers(GLsizei n, const GLuint *framebuffers) {
    GL_CHECK(glDeleteFramebuffers(n, framebuffers));
}

void GL::deleteRenderbuffers(GLsizei n, const GLuint *renderbuffers) {
    GL_CHECK(glDeleteRenderbuffers(n, renderbuffers));
}

GLenum GL::checkFramebufferStatus(GLenum target) {
    GLenum status = glCheckFramebufferStatus(target);
    GL_CHECK({});
    return status;
}

void GL::blitFramebuffer(GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1,
                         GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1,
                         GLbitfield mask, GLenum filter) {
    GL_CHECK(glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter));
}

// GPU timestamp queries for debug/profiler.cpp. Only wired up for TANGRAM_LINUX for now: the
// desktop ARB_timer_query entry points are exported by GLVND libOpenGL and Mesa dispatches them
// on GLES contexts exposing GL_EXT_disjoint_timer_query as well (shared dispatch slots). Other
// platforms report no support, making Profiler::GpuZone a silent no-op there.
#if defined(TANGRAM_LINUX)

#ifndef GL_GPU_DISJOINT_EXT
#define GL_GPU_DISJOINT_EXT 0x8FBB
#endif

bool GL::timerQueriesSupported() {
    // Runtime probe instead of an extension-string check: GL::getString(GL_EXTENSIONS) is
    // faked on desktop builds (see above), and this works for both a GLES context with
    // GL_EXT_disjoint_timer_query and a desktop GL >= 3.3 context.
    static int s_supported = -1;
    if (s_supported < 0) {
        while (glGetError() != GL_NO_ERROR) {}  // clear any pending errors
        GLuint q = 0;
        GLint64 t = 0;
        glGenQueries(1, &q);
        if (q) {
            glQueryCounter(q, GL_TIMESTAMP);
            glGetInteger64v(GL_TIMESTAMP, &t);
        }
        s_supported = (q && t > 0 && glGetError() == GL_NO_ERROR) ? 1 : 0;
        if (q) { glDeleteQueries(1, &q); }
    }
    return s_supported == 1;
}

void GL::genQueries(GLsizei n, GLuint *ids) {
    GL_CHECK(glGenQueries(n, ids));
}

void GL::deleteQueries(GLsizei n, const GLuint *ids) {
    GL_CHECK(glDeleteQueries(n, ids));
}

void GL::queryCounter(GLuint id) {
    GL_CHECK(glQueryCounter(id, GL_TIMESTAMP));
}

bool GL::queryResultAvailable(GLuint id) {
    GLuint available = 0;
    GL_CHECK(glGetQueryObjectuiv(id, GL_QUERY_RESULT_AVAILABLE, &available));
    return available != 0;
}

unsigned long long GL::queryResult64(GLuint id) {
    GLuint64 result = 0;
    GL_CHECK(glGetQueryObjectui64v(id, GL_QUERY_RESULT, &result));
    return result;
}

long long GL::gpuTimestampNs() {
    GLint64 t = 0;
    GL_CHECK(glGetInteger64v(GL_TIMESTAMP, &t));
    return t;
}

bool GL::gpuDisjointOccurred() {
    GLint disjoint = 0;
    glGetIntegerv(GL_GPU_DISJOINT_EXT, &disjoint);
    while (glGetError() != GL_NO_ERROR) {}  // enum only exists with GL_EXT_disjoint_timer_query
    return disjoint != 0;
}

#else  // !TANGRAM_LINUX

bool GL::timerQueriesSupported() { return false; }
void GL::genQueries(GLsizei n, GLuint *ids) { for (GLsizei i = 0; i < n; i++) { ids[i] = 0; } }
void GL::deleteQueries(GLsizei, const GLuint *) {}
void GL::queryCounter(GLuint) {}
bool GL::queryResultAvailable(GLuint) { return false; }
unsigned long long GL::queryResult64(GLuint) { return 0; }
long long GL::gpuTimestampNs() { return 0; }
bool GL::gpuDisjointOccurred() { return false; }

#endif

}
