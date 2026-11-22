<script setup>
import { computed, onMounted, ref } from 'vue'
import PageHeader from '../components/PageHeader.vue'
import OreArt from '../components/OreArt.vue'
import api, { auth } from '../api'

const cards = ref([])
const unlocked = ref([])
const active = ref(null)
const loading = ref(true)
const error = ref('')

const loggedIn = computed(() => Boolean(auth.token))
const progress = computed(() => {
  if (!cards.value.length) return 0
  return Math.round((unlocked.value.length / cards.value.length) * 100)
})

function isUnlocked(card) {
  return loggedIn.value && unlocked.value.includes(card.key)
}

onMounted(async () => {
  try {
    cards.value = await api.ores()
    if (loggedIn.value) {
      const data = await api.collection()
      unlocked.value = data.unlocked
    }
  } catch (e) {
    error.value = e?.response?.data?.detail || '图鉴加载失败'
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <PageHeader />

  <main class="page">
    <h1 class="page-title">矿石图鉴</h1>
    <p class="page-subtitle">
      识别过的矿石会在这里点亮。登录后开始收集你的 5 张矿石知识卡吧。
    </p>

    <div class="card gallery-progress">
      <strong style="white-space: nowrap">
        {{ loggedIn ? '收集进度' : '登录后可收集' }}
      </strong>
      <div class="progress-track">
        <i :style="{ width: `${loggedIn ? progress : 0}%` }" />
      </div>
      <strong style="white-space: nowrap">
        {{ loggedIn ? `${unlocked.length} / ${cards.length}` : '—' }}
      </strong>
    </div>

    <p v-if="error" class="error-text">{{ error }}</p>
    <p v-if="loading" class="muted">正在加载图鉴…</p>

    <div class="gallery-grid">
      <article
        v-for="(card, i) in cards"
        :key="card.key"
        class="card ore-card"
        :class="{ locked: loggedIn && !isUnlocked(card) }"
        @click="active = card"
      >
        <OreArt :color="card.color" :variant="i" />
        <h3>{{ card.name }}</h3>
        <div class="pinyin">{{ card.pinyin }}</div>
        <span class="state">
          {{ loggedIn ? (isUnlocked(card) ? '✓ 已点亮' : '待点亮') : '点击查看' }}
        </span>
      </article>
    </div>
  </main>

  <div v-if="active" class="modal-mask" @click.self="active = null">
    <div class="modal">
      <div style="display: flex; gap: 16px; align-items: center">
        <OreArt :color="active.color" :variant="0" style="width: 110px; flex: none" />
        <div>
          <h3>{{ active.emoji }} {{ active.name }}</h3>
          <p class="modal-sub" style="margin: 4px 0 0">{{ active.pinyin }}</p>
        </div>
      </div>

      <p style="line-height: 1.8; color: var(--ink-soft); font-size: 14px">
        {{ active.fun_fact }}
      </p>

      <div class="detail-grid">
        <div class="detail-item">
          <span>莫氏硬度</span>
          <strong>{{ active.hardness }}</strong>
        </div>
        <div class="detail-item">
          <span>光泽</span>
          <strong>{{ active.luster }}</strong>
        </div>
        <div class="detail-item">
          <span>条痕</span>
          <strong>{{ active.streak }}</strong>
        </div>
        <div class="detail-item">
          <span>图鉴状态</span>
          <strong>{{ isUnlocked(active) ? '已点亮' : loggedIn ? '待点亮' : '未登录' }}</strong>
        </div>
      </div>

      <div class="modal-actions">
        <button class="btn btn-primary" @click="active = null">知道啦</button>
      </div>
    </div>
  </div>
</template>
