<script setup>
import { computed, onMounted, ref } from 'vue'
import api from '../api'

const props = defineProps({
  recordId: { type: Number, required: true },
  predLabel: { type: String, default: '' }
})
const emit = defineEmits(['close', 'done'])

const cards = ref([])
const selected = ref('')
const loading = ref(false)
const error = ref('')

const options = computed(() =>
  cards.value.filter((item) => item.key !== props.predLabel)
)

onMounted(async () => {
  try {
    cards.value = await api.ores()
  } catch (e) {
    error.value = e?.response?.data?.detail || '知识卡加载失败'
  }
})

async function submit() {
  if (!selected.value) {
    error.value = '先选一个正确的答案吧'
    return
  }
  loading.value = true
  error.value = ''
  try {
    await api.badcase({
      record_id: props.recordId,
      correct_label: selected.value
    })
    emit('done', selected.value)
  } catch (e) {
    error.value = e?.response?.data?.detail || '提交失败，请稍后再试'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="modal-mask" @click.self="emit('close')">
    <div class="modal">
      <h3>识别错了？</h3>
      <p class="modal-sub">
        当前识别结果：<strong>{{ predLabel }}</strong>。选择你认为正确的矿石类别，
        管理员会在后台看到这条纠错。
      </p>

      <div class="field">
        <label>正确类别</label>
        <select v-model="selected">
          <option value="" disabled>请选择</option>
          <option v-for="item in options" :key="item.key" :value="item.key">
            {{ item.emoji }} {{ item.name }}
          </option>
        </select>
      </div>

      <p v-if="error" class="error-text">{{ error }}</p>

      <div class="modal-actions">
        <button class="btn btn-ghost" @click="emit('close')">取消</button>
        <button class="btn btn-primary" :disabled="loading" @click="submit">
          {{ loading ? '提交中…' : '提交纠错' }}
        </button>
      </div>
    </div>
  </div>
</template>
