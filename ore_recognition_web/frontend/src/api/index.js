import axios from 'axios'

const request = axios.create({ baseURL: '/api', timeout: 30000 })

// 每次请求自动携带 Token，页面里不用重复写
request.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

request.interceptors.response.use(
  (response) => response.data,
  (error) => Promise.reject(error)
)

export const auth = {
  get token() {
    return localStorage.getItem('token')
  },
  get user() {
    try {
      return JSON.parse(localStorage.getItem('user') || 'null')
    } catch {
      return null
    }
  },
  save(token, user) {
    localStorage.setItem('token', token)
    localStorage.setItem('user', JSON.stringify(user))
  },
  clear() {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
  }
}

const api = {
  health: () => request.get('/health'),

  // 知识卡
  ores: () => request.get('/ores'),
  ore: (key) => request.get(`/ores/${encodeURIComponent(key)}`),

  // 识别
  predict: (file) => {
    const form = new FormData()
    form.append('file', file)
    // 不要手写 Content-Type，浏览器/Axios 需要自动补上 multipart 的 boundary
    return request.post('/predict', form)
  },

  // 认证
  register: (payload) => request.post('/auth/register', payload),
  login: (payload) => request.post('/auth/login', payload),
  me: () => request.get('/auth/me'),

  // 个人数据
  records: (limit = 30) => request.get('/records', { params: { limit } }),
  collection: () => request.get('/collection'),
  stats: () => request.get('/stats'),
  badcase: (payload) => request.post('/badcase', payload),

  // 管理端
  admin: {
    overview: () => request.get('/admin/overview'),
    users: (params) => request.get('/admin/users', { params }),
    records: (params) => request.get('/admin/records', { params }),
    badcases: (params) => request.get('/admin/badcases', { params }),
    handleBadcase: (id, handled) =>
      request.patch(`/admin/badcases/${id}`, { handled }),
    updateOre: (key, payload) =>
      request.put(`/admin/ores/${encodeURIComponent(key)}`, payload)
  }
}

export default api
