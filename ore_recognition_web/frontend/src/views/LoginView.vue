<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import PageHeader from '../components/PageHeader.vue'
import OreArt from '../components/OreArt.vue'
import api, { auth } from '../api'

const router = useRouter()

const mode = ref('login')
const username = ref('')
const password = ref('')
const nickname = ref('')
const loading = ref(false)
const error = ref('')

const currentUser = ref(auth.user)

const title = computed(() => (mode.value === 'login' ? '欢迎回来' : '加入识矿乐园'))

async function submit() {
  error.value = ''
  if (!username.value.trim() || !password.value) {
    error.value = '请填写用户名和密码'
    return
  }
  if (mode.value === 'register' && password.value.length < 6) {
    error.value = '密码至少 6 位'
    return
  }

  loading.value = true
  try {
    const payload = {
      username: username.value.trim(),
      password: password.value
    }
    const data =
      mode.value === 'login'
        ? await api.login(payload)
        : await api.register({
            ...payload,
            nickname: nickname.value.trim() || '小小矿物学家'
          })

    auth.save(data.token, {
      username: data.username,
      nickname: data.nickname,
      is_admin: data.is_admin
    })
    router.push(data.is_admin ? '/admin' : '/gallery')
  } catch (e) {
    error.value = e?.response?.data?.detail || '操作失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

function switchMode(next) {
  mode.value = next
  error.value = ''
}

function logout() {
  auth.clear()
  currentUser.value = null
  router.push('/')
}

onMounted(() => {
  currentUser.value = auth.user
})
</script>

<template>
  <PageHeader />

  <main class="login-wrap">
    <section class="card login-card">
      <template v-if="currentUser">
        <div style="display: flex; align-items: center; gap: 16px">
          <OreArt color="#7ec7b9" :variant="1" style="width: 92px" />
          <div>
            <h1 style="font-size: 24px">你已经登录啦</h1>
            <p class="muted" style="margin: 4px 0 0">
              {{ currentUser.nickname }} · {{ currentUser.username }}
            </p>
          </div>
        </div>
        <div class="modal-actions" style="justify-content: flex-start; margin-top: 24px">
          <router-link to="/history" class="btn btn-primary">我的记录</router-link>
          <router-link
            v-if="currentUser.is_admin"
            to="/admin"
            class="btn btn-ghost"
          >
            管理端
          </router-link>
          <button class="btn btn-ghost" @click="logout">退出登录</button>
        </div>
      </template>

      <template v-else>
        <h1>{{ title }}</h1>
        <p class="muted" style="margin: 4px 0 0; font-size: 13px">
          登录后可以保存识别记录、点亮图鉴、提交纠错。
        </p>

        <div class="tabs">
          <button :class="{ active: mode === 'login' }" @click="switchMode('login')">
            登录
          </button>
          <button :class="{ active: mode === 'register' }" @click="switchMode('register')">
            注册
          </button>
        </div>

        <div class="field">
          <label>用户名</label>
          <input v-model="username" placeholder="例如 admin" autocomplete="username" />
        </div>

        <div v-if="mode === 'register'" class="field">
          <label>昵称（可选）</label>
          <input v-model="nickname" placeholder="小小矿物学家" maxlength="20" />
        </div>

        <div class="field">
          <label>密码</label>
          <input
            v-model="password"
            type="password"
            placeholder="至少 6 位"
            autocomplete="current-password"
            @keyup.enter="submit"
          />
        </div>

        <p v-if="error" class="error-text">{{ error }}</p>

        <button class="btn btn-primary" style="width: 100%; margin-top: 8px" :disabled="loading" @click="submit">
          {{ loading ? '请稍候…' : mode === 'login' ? '登录' : '注册并登录' }}
        </button>

        <p class="muted" style="font-size: 12px; margin: 16px 0 0; line-height: 1.7">
          课程演示管理员：<strong>admin / 123456</strong><br />
          普通用户可自行注册。
        </p>
      </template>
    </section>
  </main>
</template>
