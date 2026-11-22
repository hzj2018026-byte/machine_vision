<script setup>
import { computed, onMounted, ref } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import CorrectionDialog from '../components/CorrectionDialog.vue'
import api, { auth } from '../api'
import { formatPercent, formatTime } from '../utils/format'

const stats = ref(null)
const records = ref([])
const activeRecord = ref(null)
const corrected = ref({})
const loading = ref(true)
const error = ref('')

const loggedIn = computed(() => Boolean(auth.token))

const maxDay = computed(() => {
  const counts = (stats.value?.days || []).map((day) => day.count)
  return Math.max(1, ...counts)
})

function barHeight(count) {
  if (!count) return '6px'
  return `${Math.max(10, Math.round((count / maxDay.value) * 100))}%`
}

async function load() {
  if (!loggedIn.value) {
    loading.value = false
    return
  }
  try {
    const [statData, recordData] = await Promise.all([
      api.stats(),
      api.records(30)
    ])
    stats.value = statData
    records.value = recordData
  } catch (e) {
    error.value = e?.response?.data?.detail || '加载记录失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

function onCorrected(label) {
  if (activeRecord.value) {
    corrected.value[activeRecord.value.id] = label
  }
  activeRecord.value = null
}

onMounted(load)
</script>

<template>
  <PageHeader />

  <main class="page">
    <h1 class="page-title">我的记录</h1>
    <p class="page-subtitle">看看你认识了多少种矿石，哪一天观察得最认真。</p>

    <div v-if="!loggedIn" class="card panel" style="text-align: center; padding: 50px 24px">
      <div style="font-size: 48px">🔐</div>
      <h3 style="margin: 12px 0 6px">登录后才能保存识别记录</h3>
      <p class="muted" style="margin: 0 0 20px">
        登录后，每次识别都会记录在这里，还能解锁图鉴、提交纠错。
      </p>
      <router-link to="/login" class="btn btn-primary">去登录 / 注册</router-link>
    </div>

    <template v-else>
      <p v-if="error" class="error-text">{{ error }}</p>

      <div v-if="stats" class="stats-grid">
        <div class="card stat-card">
          <div class="value">{{ stats.total }}</div>
          <div class="label">总识别数</div>
        </div>
        <div class="card stat-card">
          <div class="value">{{ stats.week_total }}</div>
          <div class="label">近 7 天识别</div>
        </div>
        <div class="card stat-card">
          <div class="value">{{ formatPercent(stats.avg_confidence) }}</div>
          <div class="label">平均把握度</div>
        </div>
        <div class="card stat-card">
          <div class="value">{{ stats.unlocked }} / {{ stats.ore_total }}</div>
          <div class="label">图鉴收集</div>
        </div>
      </div>

      <section v-if="stats" class="card panel">
        <div class="panel-title">
          <h3>近 7 天识别</h3>
          <span class="chip">共 {{ stats.week_total }} 次</span>
        </div>
        <div class="week-chart">
          <div v-for="day in stats.days" :key="day.date" class="week-col">
            <span class="count">{{ day.count }}</span>
            <div class="track">
              <i :class="{ empty: !day.count }" :style="{ height: barHeight(day.count) }" />
            </div>
            <span class="day">{{ day.date }}</span>
          </div>
        </div>
      </section>

      <section v-if="stats" class="card panel">
        <div class="panel-title">
          <h3>类别分布</h3>
        </div>
        <div v-if="stats.labels.length" class="chips">
          <span v-for="item in stats.labels" :key="item.label" class="chip">
            {{ item.label }} × {{ item.count }}
          </span>
        </div>
        <p v-else class="muted" style="margin: 0">还没有识别记录。</p>
      </section>

      <section class="card panel">
        <div class="panel-title">
          <h3>历史记录</h3>
          <span class="chip">最近 {{ records.length }} 条</span>
        </div>

        <div v-if="records.length" class="record-list">
          <article v-for="record in records" :key="record.id" class="record-row">
            <img :src="record.image_path" :alt="record.label" />
            <div>
              <h4>{{ record.label }}</h4>
              <p>
                {{ formatTime(record.created_at) }} · 置信度
                {{ formatPercent(record.confidence) }}
              </p>
              <p v-if="corrected[record.id]" class="corrected-tag">
                已纠正为 {{ corrected[record.id] }}
              </p>
            </div>
            <div class="right">
              <span class="chip">{{ record.confidence >= 0.6 ? '清晰' : '有点模糊' }}</span>
              <button class="btn btn-ghost" style="padding: 6px 12px" @click="activeRecord = record">
                识别错了？
              </button>
            </div>
          </article>
        </div>
        <p v-else class="muted" style="margin: 0">
          还没有识别记录，去 <router-link to="/shoot" style="color: var(--brand-deep); font-weight: 700">拍一拍</router-link> 试试吧。
        </p>
      </section>
    </template>

    <CorrectionDialog
      v-if="activeRecord"
      :record-id="activeRecord.id"
      :pred-label="activeRecord.label"
      @close="activeRecord = null"
      @done="onCorrected"
    />
  </main>
</template>
