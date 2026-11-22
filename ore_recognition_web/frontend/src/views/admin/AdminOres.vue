<script setup>
import { onMounted, ref } from 'vue'
import api from '../../api'
import OreArt from '../../components/OreArt.vue'

const cards = ref([])
const loading = ref(true)
const error = ref('')
const editing = ref(null)
const form = ref(null)
const saving = ref(false)
const formError = ref('')
const toast = ref('')

async function load() {
  loading.value = true
  try {
    cards.value = await api.ores()
  } catch (e) {
    error.value = e?.response?.data?.detail || '知识卡加载失败'
  } finally {
    loading.value = false
  }
}

function openEdit(card) {
  editing.value = card.key
  form.value = { ...card }
  formError.value = ''
}

function closeEdit() {
  editing.value = null
  form.value = null
  formError.value = ''
}

async function save() {
  if (!form.value.name.trim() || !form.value.emoji.trim()) {
    formError.value = '名称和图标不能为空'
    return
  }
  saving.value = true
  formError.value = ''
  try {
    const payload = { ...form.value }
    delete payload.key
    const updated = await api.admin.updateOre(editing.value, payload)
    const index = cards.value.findIndex((card) => card.key === updated.key)
    if (index >= 0) cards.value[index] = updated
    closeEdit()
    toast.value = '已保存，前台图鉴会同步更新'
    setTimeout(() => {
      toast.value = ''
    }, 2600)
  } catch (e) {
    formError.value = e?.response?.data?.detail || '保存失败'
  } finally {
    saving.value = false
  }
}

onMounted(load)
</script>

<template>
  <p v-if="error" class="error-text">{{ error }}</p>
  <p v-if="loading" class="muted">正在加载知识卡…</p>
  <p v-if="toast" style="color: var(--brand-deep); font-weight: 700">
    ✓ {{ toast }}
  </p>

  <div v-if="!loading" class="ore-admin-grid">
    <article v-for="(card, i) in cards" :key="card.key" class="ore-admin-card">
      <div class="top">
        <div>
          <h4>{{ card.emoji }} {{ card.name }}</h4>
          <p style="margin: 2px 0 0; color: var(--ink-soft); font-size: 11px">
            {{ card.pinyin }}
          </p>
        </div>
        <OreArt :color="card.color" :variant="i" style="width: 58px; flex: none" />
      </div>
      <p>{{ card.fun_fact }}</p>
      <div class="chips" style="margin-bottom: 12px">
        <span class="chip">硬度 {{ card.hardness }}</span>
        <span class="chip">{{ card.luster }}</span>
      </div>
      <button class="btn btn-ghost" style="padding: 7px 14px" @click="openEdit(card)">
        编辑知识卡
      </button>
    </article>
  </div>

  <div v-if="form" class="modal-mask" @click.self="closeEdit">
    <div class="modal" style="width: min(620px, 100%)">
      <h3>编辑「{{ form.name }}」知识卡</h3>
      <p class="modal-sub">保存后前台图鉴会立即同步。</p>

      <div class="ore-edit-grid">
        <div class="field">
          <label>名称</label>
          <input v-model="form.name" maxlength="20" />
        </div>
        <div class="field">
          <label>拼音</label>
          <input v-model="form.pinyin" maxlength="50" />
        </div>
        <div class="field">
          <label>图标 Emoji</label>
          <input v-model="form.emoji" maxlength="8" />
        </div>
        <div class="field">
          <label>主题色</label>
          <div style="display: flex; gap: 8px">
            <input v-model="form.color" placeholder="#2f8f83" />
            <input type="color" v-model="form.color" style="width: 54px; padding: 4px" />
          </div>
        </div>
        <div class="field">
          <label>莫氏硬度</label>
          <input v-model="form.hardness" maxlength="50" />
        </div>
        <div class="field">
          <label>光泽</label>
          <input v-model="form.luster" maxlength="100" />
        </div>
        <div class="field">
          <label>条痕</label>
          <input v-model="form.streak" maxlength="100" />
        </div>
        <div class="field full">
          <label>小知识</label>
          <textarea v-model="form.fun_fact" />
        </div>
      </div>

      <p v-if="formError" class="error-text">{{ formError }}</p>

      <div class="modal-actions">
        <button class="btn btn-ghost" @click="closeEdit">取消</button>
        <button class="btn btn-primary" :disabled="saving" @click="save">
          {{ saving ? '保存中…' : '保存修改' }}
        </button>
      </div>
    </div>
  </div>
</template>
