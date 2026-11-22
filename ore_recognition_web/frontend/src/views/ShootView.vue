<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { useRouter } from 'vue-router'
import PageHeader from '../components/PageHeader.vue'
import OreArt from '../components/OreArt.vue'
import CorrectionDialog from '../components/CorrectionDialog.vue'
import api, { auth } from '../api'
import { formatPercent } from '../utils/format'

const router = useRouter()

const fileInput = ref(null)
const preview = ref('')
const loading = ref(false)
const error = ref('')
const result = ref(null)
const showCorrection = ref(false)
const corrected = ref('')

const loggedIn = computed(() => Boolean(auth.token))

function pick() {
  fileInput.value?.click()
}

async function onFile(event) {
  const file = event.target.files?.[0]
  if (!file) return

  // 换图前释放旧的本地预览地址
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = URL.createObjectURL(file)
  result.value = null
  corrected.value = ''
  error.value = ''

  await recognize(file)
  event.target.value = ''   // 允许再次选择同一张图
}

async function recognize(file) {
  loading.value = true
  error.value = ''
  try {
    result.value = await api.predict(file)
    speak(result.value.message)
  } catch (e) {
    error.value = e?.response?.data?.detail || '识别失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

function speak(text) {
  if (!text || !window.speechSynthesis) return
  window.speechSynthesis.cancel()
  const utter = new SpeechSynthesisUtterance(text)
  utter.lang = 'zh-CN'
  utter.rate = 0.95
  window.speechSynthesis.speak(utter)
}

function again() {
  result.value = null
  corrected.value = ''
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  pick()
}

function openCorrection() {
  if (!loggedIn.value) {
    router.push('/login')
    return
  }
  showCorrection.value = true
}

function onCorrected(label) {
  corrected.value = label
  showCorrection.value = false
}

onBeforeUnmount(() => {
  if (preview.value) URL.revokeObjectURL(preview.value)
  if (window.speechSynthesis) window.speechSynthesis.cancel()
})
</script>

<template>
  <PageHeader />

  <main class="page">
    <h1 class="page-title">拍一拍矿石</h1>
    <p class="page-subtitle">
      选择或拍摄一张矿石照片，系统会调用 ResNet18 模型识别。登录后识别结果会进入你的
      个人记录，并点亮矿石图鉴。
    </p>

    <div class="shoot-grid">
      <section class="card shoot-panel">
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          capture="environment"
          hidden
          @change="onFile"
        />

        <div class="dropzone" @click="pick">
          <img v-if="preview" :src="preview" alt="待识别矿石预览" />
          <div v-else class="hint">
            <div style="font-size: 46px; margin-bottom: 10px">📷</div>
            <strong>点击选择一张矿石图片</strong>
            手机端会优先打开后置摄像头，桌面端直接选文件。
            <div class="muted" style="margin-top: 8px; font-size: 12px">
              支持 JPG / PNG / WebP / BMP，最大 5MB
            </div>
          </div>
        </div>

        <div class="shoot-actions">
          <button class="btn btn-primary" :disabled="loading" @click="pick">
            {{ preview ? '换一张图' : '选择 / 拍摄图片' }}
          </button>
          <button
            v-if="preview"
            class="btn btn-ghost"
            :disabled="loading"
            @click="again"
          >
            重新拍摄
          </button>
          <button
            v-if="result && loggedIn"
            class="btn btn-ghost"
            @click="openCorrection"
          >
            识别错了？
          </button>
        </div>

        <div v-if="loading" class="loading-line">
          <span class="spinner" /> 正在识别，请稍候…
        </div>
        <p v-if="error" class="error-text" style="margin-top: 14px">{{ error }}</p>
        <p v-if="corrected" class="corrected-tag" style="margin-top: 14px">
          已提交纠错：{{ corrected }}
        </p>
      </section>

      <section class="card result-panel">
        <template v-if="result">
          <div class="result-head">
            <div>
              <span
                class="badge"
                :class="result.uncertain ? 'badge-warn' : 'badge-ok'"
              >
                {{ result.uncertain ? '不太确定' : '识别成功' }}
              </span>
              <h2 class="result-label">{{ result.label }}</h2>
              <p class="result-message">{{ result.message }}</p>
            </div>
            <div class="result-emoji">{{ result.card?.emoji || '💎' }}</div>
          </div>

          <div class="top3">
            <div v-for="item in result.top3" :key="item.label" class="top3-row">
              <span>{{ item.label }}</span>
              <div class="top3-track">
                <i :style="{ width: `${Math.max(4, item.confidence * 100)}%` }" />
              </div>
              <strong>{{ formatPercent(item.confidence) }}</strong>
            </div>
          </div>

          <div v-if="result.card" class="knowledge">
            <h4>{{ result.card.emoji }} {{ result.card.name }}小知识</h4>
            <p>{{ result.card.fun_fact }}</p>
            <div class="meta-grid">
              <span class="chip">硬度 {{ result.card.hardness }}</span>
              <span class="chip">{{ result.card.luster }}</span>
              <span class="chip">条痕 {{ result.card.streak }}</span>
            </div>
          </div>
        </template>

        <div v-else class="placeholder">
          <div>
            <OreArt color="#d9c8a8" :variant="1" style="width: 150px; margin: 0 auto 18px" />
            <div class="big">🔍</div>
            <strong>识别结果会显示在这里</strong>
            <p style="max-width: 330px; margin: 8px auto 0; line-height: 1.7">
              尽量只放一块矿石、保持光线充足，识别会更准确。
            </p>
          </div>
        </div>
      </section>
    </div>

    <CorrectionDialog
      v-if="showCorrection && result"
      :record-id="result.record_id"
      :pred-label="result.label"
      @close="showCorrection = false"
      @done="onCorrected"
    />
  </main>
</template>
