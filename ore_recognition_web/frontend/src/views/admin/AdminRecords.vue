<script setup>
import { onMounted, ref } from 'vue'
import api from '../../api'
import AdminPager from '../../components/AdminPager.vue'
import { formatPercent, formatTime } from '../../utils/format'

const ores = ref([])
const label = ref('')
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
    const data = await api.admin.records({
      label: label.value,
      page: target,
      page_size: pageSize
    })
    items.value = data.items
    total.value = data.total
    page.value = target
  } catch (e) {
    error.value = e?.response?.data?.detail || '加载识别记录失败'
  } finally {
    loading.value = false
  }
}

onMounted(async () => {
  try {
    ores.value = await api.ores()
  } catch {
    ores.value = []
  }
  load(1)
})
</script>

<template>
  <div class="admin-panel">
    <div class="admin-toolbar">
      <select v-model="label" @change="load(1)">
        <option value="">全部类别</option>
        <option v-for="ore in ores" :key="ore.key" :value="ore.key">
          {{ ore.name }}
        </option>
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
            <th>预测类别</th>
            <th>置信度</th>
            <th>时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="record in items" :key="record.id">
            <td>{{ record.id }}</td>
            <td><img :src="record.image_path" :alt="record.label" /></td>
            <td>{{ record.username || '游客' }}</td>
            <td><strong>{{ record.label }}</strong></td>
            <td>{{ formatPercent(record.confidence) }}</td>
            <td>{{ formatTime(record.created_at) }}</td>
          </tr>
          <tr v-if="!items.length">
            <td colspan="6" class="muted">没有识别记录</td>
          </tr>
        </tbody>
      </table>
    </div>

    <AdminPager :page="page" :page-size="pageSize" :total="total" @change="load" />
  </div>
</template>
