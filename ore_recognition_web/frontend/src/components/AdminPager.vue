<script setup>
import { computed } from 'vue'

const props = defineProps({
  page: { type: Number, default: 1 },
  pageSize: { type: Number, default: 10 },
  total: { type: Number, default: 0 }
})
const emit = defineEmits(['change'])

const totalPages = computed(() => Math.max(1, Math.ceil(props.total / props.pageSize)))

function go(target) {
  if (target < 1 || target > totalPages.value || target === props.page) return
  emit('change', target)
}
</script>

<template>
  <div class="admin-pager">
    <span>共 {{ total }} 条 · 第 {{ page }} / {{ totalPages }} 页</span>
    <button :disabled="page <= 1" @click="go(page - 1)">上一页</button>
    <button :disabled="page >= totalPages" @click="go(page + 1)">下一页</button>
  </div>
</template>
