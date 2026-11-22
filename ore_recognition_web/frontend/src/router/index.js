import { createRouter, createWebHistory } from 'vue-router'
import { auth } from '../api'

import HomeView from '../views/HomeView.vue'
import ShootView from '../views/ShootView.vue'
import GalleryView from '../views/GalleryView.vue'
import HistoryView from '../views/HistoryView.vue'
import LoginView from '../views/LoginView.vue'
import AdminView from '../views/AdminView.vue'

const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/shoot', name: 'shoot', component: ShootView },
  { path: '/gallery', name: 'gallery', component: GalleryView },
  { path: '/history', name: 'history', component: HistoryView },
  { path: '/login', name: 'login', component: LoginView },
  { path: '/admin', name: 'admin', component: AdminView }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 })
})

router.beforeEach((to) => {
  if (to.path.startsWith('/admin') && !auth.user?.is_admin) {
    return { path: '/login' }
  }
})

export default router
