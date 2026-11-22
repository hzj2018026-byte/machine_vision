<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { auth } from '../api'

const route = useRoute()
const router = useRouter()

const user = computed(() => auth.user)
const loggedIn = computed(() => Boolean(auth.token))

const links = computed(() => {
  const items = [
    { path: '/', label: '首页' },
    { path: '/shoot', label: '拍一拍' },
    { path: '/gallery', label: '矿石图鉴' }
  ]
  if (loggedIn.value) items.push({ path: '/history', label: '我的记录' })
  if (user.value?.is_admin) items.push({ path: '/admin', label: '管理端' })
  return items
})

function isActive(path) {
  return path === '/' ? route.path === '/' : route.path.startsWith(path)
}

function logout() {
  auth.clear()
  router.push('/')
}
</script>

<template>
  <header class="topbar">
    <div class="topbar-inner">
      <router-link to="/" class="brand">
        <span class="brand-mark">💎</span>
        <span>识矿乐园</span>
      </router-link>

      <nav class="nav-links">
        <router-link
          v-for="item in links"
          :key="item.path"
          :to="item.path"
          :class="{ active: isActive(item.path) }"
        >
          {{ item.label }}
        </router-link>
      </nav>

      <div v-if="loggedIn" class="nav-user">
        <span class="avatar">⛏️</span>
        <div>
          <strong>{{ user?.nickname || '小小矿物学家' }}</strong>
          <small>{{ user?.username }}</small>
        </div>
        <button class="btn btn-ghost" style="padding: 8px 14px" @click="logout">
          退出
        </button>
      </div>
      <router-link v-else to="/login" class="btn btn-primary" style="padding: 9px 18px">
        登录 / 注册
      </router-link>
    </div>
  </header>
</template>
