<script setup>
import { computed, onMounted, ref } from 'vue'
import api from '../../api'
import { formatPercent } from '../../utils/format'

const data = ref(null)
const loading = ref(true)
const error = ref('')

const maxDay = computed(() => {
  const counts = (data.value?.days || []).map((day) => day.count)
  return Math.max(1, ...counts)
})

function barHeight(count) {
  if (!count) return '6px'
  return `${Math.max(10, Math.round((count / maxDay.value) * 100))}%`
}

onMounted(async () => {
  try {
    data.value = await api.admin.overview()
  } catch (e) {
    error.value = e?.response?.data?.detail || '概览加载失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <p v-if="error" class="error-text">{{ error }}</p>
  <p v-if="loading" class="muted">正在加载概览…</p>

  <template v-if="data">
    <div class="admin-grid">
      <div class="admin-stat">
        <div class="value">{{ data.user_total }}</div>
        <div class="label">注册用户</div>
      </div>
      <div class="admin-stat">
        <div class="value">{{ data.record_total }}</div>
        <div class="label">识别总数</div>
      </div>
      <div class="admin-stat">
        <div class="value">{{ data.today_records }}</div>
        <div class="label">今日识别</div>
      </div>
      <div class="admin-stat">
        <div class="value">{{ data.pending_badcases }}</div>
        <div class="label">待处理纠错</div>
      </div>
    </div>

    <div class="admin-panel">
      <div class="admin-panel-head">
        <h3>近 7 天识别趋势</h3>
        <span class="chip">平均置信度 {{ formatPercent(data.avg_confidence) }}</span>
      </div>
      <div class="week-chart">
        <div v-for="day in data.days" :key="day.date" class="week-col">
          <span class="count">{{ day.count }}</span>
          <div class="track">
            <i :class="{ empty: !day.count }" :style="{ height: barHeight(day.count) }" />
          </div>
          <span class="day">{{ day.date }}</span>
        </div>
      </div>
    </div>

    <div class="admin-panel">
      <div class="admin-panel-head">
        <h3>识别类别分布</h3>
      </div>
      <div v-if="data.labels.length" class="chips">
        <span v-for="item in data.labels" :key="item.label" class="chip">
          {{ item.label }} × {{ item.count }}
        </span>
      </div>
      <p v-else class="muted" style="margin: 0">还没有识别记录。</p>
    </div>
  </template>
</template>
