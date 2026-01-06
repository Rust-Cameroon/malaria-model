/**
 * Application route configuration that defines all routes and their protection levels.
 * This file centralizes the routing structure, making it easy to manage:
 * - Public routes accessible to all users
 * - Protected routes with role-based access control
 * - Admin routes requiring specific permissions
 * - Error routes for handling 404s and unauthorized access
 */
import { lazy } from "react";
const HomePage = lazy(() => import("../pages/welcome/HomePage"));
const DemoPage = lazy(() => import("../pages/demo/DemoPage"));
const AnalyzePage = lazy(() => import("../pages/analyze/AnalyzePage"));

const routes = [
  {
    path: "/",
    component: HomePage,
    isPublic: true,
  },
  {
    path: "/demo",
    component: DemoPage,
    isPublic: true,
  },
  {
    path: "/analyze",
    component: AnalyzePage,
    isPublic: true,
  },
];

export default routes;
