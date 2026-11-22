<script setup>
import { onMounted, ref } from 'vue'
import api from '../../api'
import AdminPager from '../../components/AdminPager.vue'
import { formatTime } from '../../utils/format'

const filter = ref('pending')
const items = ref([])
const total = ref(0)
const page = ref(1)
const pageSize = 10
const loading = ref(true)
const error = ref('')
const savingId = ref(null)

async function load(target = 1) {
  loading.value = true
  error.value = ''
  try {
    const params = { page: target, page_size: pageSize }
    if (filter.value === 'pending') params.handled = false
    if (filter.value === 'done') params.handled = true

    const data = await api.admin.badcases(params)
    items.value = data.items
    total.value = data.total
    page.value = target
  } catch (e) {
    error.value = e?.response?.data?.detail || '加载纠错失败'
  } finally {
    loading.value = false
  }
}

async function toggle(row) {
  savingId.value = row.id
  try {
    await api.admin.handleBadcase(row.id, !row.handled)
    // 当前页只剩一条时，处理后自动回退一页
    const nextPage =
      items.value.length === 1 && page.value > 1 ? page.value - 1 : page.value
    await load(nextPage)
  } catch (e) {
    error.value = e?.response?.data?.detail || '操作失败'
  } finally {
    savingId.value = null
  }
}

onMounted(() => load(1))
</script>

<template>
  <div class="admin-panel">
    <div class="admin-toolbar">
      <select v-model="filter" @change="load(1)">
        <option value="pending">待处理</option>
        <option value="done">已处理</option>
        <option value="all">全部</option>
      </select>
      <button class="btn btn-ghost" style="padding: 9px 18px" @click="load(1)">
        刷新
      </button>
    </div>

    <p v-if="error" class="error-text">{{ error }}</p>
    <p v-if="loading" class="muted">正在加载…</p>

    <div v-else class="admin-table-wrap">
      <table class="admin-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>图片</th>
            <th>用户</th>
            <th>原预测</th>
            <th>纠正为</th>
            <th>时间</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in items" :key="row.id">
            <td>{{ row.id }}</td>
            <td><img :src="row.image_path" :alt="row.pred_label" /></td>
            <td>{{ row.username || '—' }}</td>
            <td>{{ row.pred_label }}</td>
            <td><strong>{{ row.correct_label }}</strong></td>
            <td>{{ formatTime(row.created_at) }}</td>
            <td>
              <span class="status-pill" :class="row.handled ? 'status-done' : 'status-pending'">
                {{ row.handled ? '已处理' : '待处理' }}
              </span>
            </td>
            <td>
              <button
                class="btn btn-ghost"
                style="padding: 6px 12px"
                :disabled="savingId === row.id"
                @click="toggle(row)"
              >
                {{ row.handled ? '恢复待处理' : '标记已处理' }}
              </button>
            </td>
          </tr>
          <tr v-if="!items.length">
            <td colspan="8" class="muted">没有纠错记录</td>
          </tr>
        </tbody>
      </table>
    </div>

    <AdminPager :page="page" :page-size="pageSize" :total="total" @change="load" />
  </div>
</template>
