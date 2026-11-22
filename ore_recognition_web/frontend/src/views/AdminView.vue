<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { auth } from '../api'
import '../styles/admin.css'

import AdminOverview from './admin/AdminOverview.vue'
import AdminUsers from './admin/AdminUsers.vue'
import AdminRecords from './admin/AdminRecords.vue'
import AdminBadcases from './admin/AdminBadcases.vue'
import AdminOres from './admin/AdminOres.vue'

const router = useRouter()
const user = auth.user

const tabs = [
  { key: 'overview', label: '数据概览', icon: '📊', subtitle: '用户、识别量、纠错与七日趋势' },
  { key: 'users', label: '用户管理', icon: '👥', subtitle: '搜索用户，查看每个人的识别次数' },
  { key: 'records', label: '识别记录', icon: '🖼️', subtitle: '全站识别记录，可按矿石类别筛选' },
  { key: 'badcases', label: '纠错反馈', icon: '🛠️', subtitle: '处理用户提交的识别纠错' },
  { key: 'ores', label: '知识卡', icon: '💎', subtitle: '编辑 5 类矿石的科普内容' }
]

const components = {
  overview: AdminOverview,
  users: AdminUsers,
  records: AdminRecords,
  badcases: AdminBadcases,
  ores: AdminOres
}

const active = ref('overview')
const currentTab = computed(() => tabs.find((tab) => tab.key === active.value))
const currentComponent = computed(() => components[active.value])

function logout() {
  auth.clear()
  router.push('/')
}
</script>

<template>
  <div class="admin-shell">
    <aside class="sidebar">
      <router-link to="/" class="brand">
        <span class="brand-mark">💎</span>
        <span class="brand-text">
          <strong>识矿乐园</strong>
          <small>管理端</small>
        </span>
      </router-link>

      <div class="who">
        <span class="avatar">👑</span>
        <div>
          <strong>{{ user?.nickname || '管理员' }}</strong>
          <small>{{ user?.username }}</small>
        </div>
      </div>

      <nav class="menu">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          :class="{ active: active === tab.key }"
          @click="active = tab.key"
        >
          <span class="icon">{{ tab.icon }}</span>{{ tab.label }}
        </button>
      </nav>

      <div class="side-foot">
        <router-link to="/" class="foot-link">← 返回前台</router-link>
        <button class="foot-link" @click="logout">退出登录</button>
      </div>
    </aside>

    <main class="admin-main">
      <div class="admin-head">
        <div>
          <h1>{{ currentTab.label }}</h1>
          <p>{{ currentTab.subtitle }}</p>
        </div>
      </div>

      <component :is="currentComponent" />
    </main>
  </div>
</template>
