<script setup>
import { onMounted, ref } from 'vue'
import api from '../../api'
import AdminPager from '../../components/AdminPager.vue'
import { formatTime } from '../../utils/format'

const keyword = ref('')
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 10
const loading = ref(true)
const error = ref('')

async function load(target = 1) {
  loading.value = true
  error.value = ''
  try {
    const data = await api.admin.users({
      query: keyword.value.trim(),
      page: target,
      page_size: pageSize
    })
    items.value = data.items
    total.value = data.total
    page.value = target
  } catch (e) {
    error.value = e?.response?.data?.detail || '加载用户失败'
  } finally {
    loading.value = false
  }
}

onMounted(() => load(1))
</script>

<template>
  <div class="admin-panel">
    <div class="admin-toolbar">
      <input
        v-model="keyword"
        placeholder="搜索用户名 / 昵称"
        @keyup.enter="load(1)"
      />
      <button class="btn btn-primary" style="padding: 9px 18px" @click="load(1)">
        搜索
      </button>
    </div>

    <p v-if="error" class="error-text">{{ error }}</p>
    <p v-if="loading" class="muted">正在加载…</p>

    <div v-else class="admin-table-wrap">
      <table class="admin-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>用户名</th>
            <th>昵称</th>
            <th>角色</th>
            <th>识别次数</th>
            <th>注册时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="user in items" :key="user.id">
            <td>{{ user.id }}</td>
            <td><strong>{{ user.username }}</strong></td>
            <td>{{ user.nickname }}</td>
            <td>
              <span class="status-pill" :class="user.is_admin ? 'status-done' : 'status-pending'">
                {{ user.is_admin ? '管理员' : '普通用户' }}
              </span>
            </td>
            <td>{{ user.record_count }}</td>
            <td>{{ formatTime(user.created_at) }}</td>
          </tr>
          <tr v-if="!items.length">
            <td colspan="6" class="muted">没有匹配的用户</td>
          </tr>
        </tbody>
      </table>
    </div>

    <AdminPager :page="page" :page-size="pageSize" :total="total" @change="load" />
  </div>
</template>
